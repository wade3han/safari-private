# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))
# Also adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py
# But we pass in the loss to avoid recomputation
import functools
from typing import Any, Dict, Optional, Callable, Literal, Union, Tuple, Sequence, List

import nltk
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
from torchmetrics.functional.text.rouge import (
    ALLOWED_ACCUMULATE_VALUES,
    ALLOWED_ROUGE_KEYS,
    _rouge_score_compute,
    _rouge_score_update,
)

from torchmetrics import Metric

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

try:
    from apex.transformer import parallel_state
except ImportError:
    parallel_state = None


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, ignore_index=-100, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = target.numel()
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)

class NumTokens(Metric):
    """Keep track of how many tokens we've seen.
    """
    # TODO: how do we prevent the reset between the epochs? The reset happens on the 1st batch
    # of the next epoch.
    # Right now the hack is that we override reset(), which would mess up the forward method.
    # We then override forward to do the right thing.

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum",
                       persistent=True)  # We want the count to be saved to state-dict
        if parallel_state is not None and not parallel_state.is_unitialized():
            self.tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            self.tensor_parallel_world_size = 1

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        self.count += target.numel() // self.tensor_parallel_world_size

    def compute(self) -> Tensor:
        return self.count

    def reset(self):
        count = self.count
        super().reset()
        self.count = count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()


class ROUGEScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(
        self,
        rouge_key: str,
        use_stemmer: bool = False,
        normalizer: Callable[[str], str] = None,
        tokenizer: Callable[[str], Sequence[str]] = None,
        accumulate: Literal["avg", "best"] = "best",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        rouge_keys = ("rouge1", "rouge2", "rougeL", "rougeLsum")
        self.rouge_keys = rouge_keys
        self.rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]

        self.rouge_key = rouge_key
        self.stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.accumulate = accumulate

        # self.add_state(f"{rouge_key}", [], dist_reduce_fx=None)
        # Adding stated dynamically to prevent IndexError during sync function as some lists can be empty.
        for rouge_key in self.rouge_keys:
            for score in ["fmeasure", "precision", "recall"]:
                self.add_state(f"{rouge_key}_{score}", [], dist_reduce_fx=None)

    def update(
        self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str], Sequence[Sequence[str]]]
    ) -> None:
        """Update state with predictions and targets."""
        if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
            target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]

        if isinstance(preds, str):
            preds = [preds]

        if isinstance(target, str):
            target = [[target]]

        output: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(
            preds,
            target,
            self.rouge_keys_values,
            stemmer=self.stemmer,
            normalizer=self.normalizer,
            tokenizer=self.tokenizer,
            accumulate=self.accumulate,
        )
        # rouge_key, score = self.rouge_key.split("_")
        # output_key = rouge_key[len('rouge'):]
        # try:
        #     output_key = int(output_key)
        # except ValueError:
        #     pass
        # metrics = output[output_key]
        # for metric in metrics:
        #     value = metric[score]
        #     getattr(self, self.rouge_key).append(value.to(self.device))
        for rouge_key, metrics in output.items():
            for metric in metrics:
                for tp, value in metric.items():
                    getattr(self, f"rouge{rouge_key}_{tp}").append(value.to(self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Calculate (Aggregate and provide confidence intervals) ROUGE score."""
        update_output = {}
        for rouge_key in self.rouge_keys_values:
            for tp in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{tp}"] = getattr(self, f"rouge{rouge_key}_{tp}")

        return _rouge_score_compute(update_output)[self.rouge_key]

    def __hash__(self) -> int:
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]
        for key in self._defaults:
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))

torchmetric_fns = {
    "perplexity": Perplexity,
    "num_tokens": NumTokens,
    "rouge1_fmeasure": functools.partial(ROUGEScore, rouge_key="rouge1_fmeasure"),
    "rouge1_precision": functools.partial(ROUGEScore, rouge_key="rouge1_precision"),
    "rouge1_recall": functools.partial(ROUGEScore, rouge_key="rouge1_recall"),
    "rouge2_fmeasure": functools.partial(ROUGEScore, rouge_key="rouge2_fmeasure"),
    "rouge2_precision": functools.partial(ROUGEScore, rouge_key="rouge2_precision"),
    "rouge2_recall": functools.partial(ROUGEScore, rouge_key="rouge2_recall"),
    "rougeL_fmeasure": functools.partial(ROUGEScore, rouge_key="rougeL_fmeasure"),
    "rougeL_precision": functools.partial(ROUGEScore, rouge_key="rougeL_precision"),
    "rougeL_recall": functools.partial(ROUGEScore, rouge_key="rougeL_recall"),
}
