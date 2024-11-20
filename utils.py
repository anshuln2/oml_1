from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from typing import Dict, Optional

import torch
import random
import math
import torch.distributed as dist
from accelerate import Accelerator
from accelerate import utils as accelerate_utils

# TODO: refactor for FSDP2
def fsdp_v1_model_params(model: FSDP):
    """
    Get all model parameters via FSDP handles
    """
    sharded_params = set()
    nonsharded_params = set()  # `NO
    for _, handle in enumerate(model._all_handles):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        target_set.add(handle.flat_param)
        yield handle.flat_param
    for _, param in model.named_parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            yield param


class FSDPModelStorage:
    """
    Storage for sharded model parameters and gradients for accumulation during TAR
    """

    def __init__(self):
        self.storage_dict = {
            "params": {},
            "grads": {},
        }

    def clear_params(self):
        self.storage_dict["params"].clear()

    def clear_grads(self):
        self.storage_dict["grads"].clear()


    def collect_param_or_grad(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        to_cpu: bool = False,
        mode: str = "grads",
        scale: float = 1.0,
    ):
        """
        Collect parameters or gradients from the FSDP model and store them efficiently.

        Args:
            model (FSDP): The FSDP model to collect from.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            to_cpu (bool): Whether to move the collected data to CPU.
            mode (str): Either "params" or "grads" to collect parameters or gradients.
            scale (float): Scaling factor for gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            # Collect parameters
            if mode == "params":
                if to_cpu:
                    self.storage_dict["params"][i] = param.detach().cpu()  # No need to clone here
                else:
                    self.storage_dict["params"][i] = param.detach()

            # Collect gradients
            if param.grad is not None and mode == "grads":
                if i not in self.storage_dict["grads"]:
                    # Create a new gradient entry in storage dict
                    self.storage_dict["grads"][i] = param.grad.detach() * scale
                else:
                    # Accumulate gradients in-place to reduce memory overhead
                    self.storage_dict["grads"][i].add_(param.grad.detach().to(self.storage_dict["grads"][i].device) * scale)

                # Move to CPU if required, but only after accumulation
                if to_cpu:
                    self.storage_dict["grads"][i] = self.storage_dict["grads"][i].cpu()

    def offload_params_or_grads(self, mode: str = "grads"):
        """
        Offload parameters or gradients from the storage to reduce memory usage.
        """
        
        if mode == "params":
            for i in self.storage_dict["params"]:
                self.storage_dict["params"][i] = self.storage_dict["params"][i].cpu()
        if mode == "grads":
            for i in self.storage_dict["grads"]:
                self.storage_dict["grads"][i] = self.storage_dict["grads"][i].cpu()
    
    def store_original_model(self, model: FSDP):
        self.original_model_params = {}
        for i, param in enumerate(fsdp_v1_model_params(model)):
            self.original_model_params[i] = param.detach().cpu()
    
    def merge_original_model(self, model: FSDP, merging_lambda: float):
        for i, param in enumerate(fsdp_v1_model_params(model)):
            param.data.copy_((1- merging_lambda) * param.data + (merging_lambda) * self.original_model_params[i].to(param.device))
            
        
    def add_from_storage_to_model(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        skip_check: bool = False,
        mode: str = "grads",
    ):
        """
        Add parameters or gradients from storage to the FSDP model.

        Args:
            model (FSDP): The FSDP model to add to.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            skip_check (bool): Whether to skip the assertion check for gradient existence.
            mode (str): Either "params" or "grads" to add parameters or gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            if mode == "params":
                param.data.copy_(self.storage_dict["params"][i].to(param.device))
            # assert either both storage and handle have grads or neither do
            if not skip_check:
                assert (i in self.storage_dict["grads"]) == (param.grad is not None)
            if i in self.storage_dict["grads"] and param.grad is not None:
                if mode == "grads":
                    param.grad += self.storage_dict["grads"][i].to(param.device)


def apply_task_vector(model: FSDP, task_vector, task_vector_coefficient: float):
    """
    Apply a task vector to the model parameters
    """
    for param, tv in zip(fsdp_v1_model_params(model), task_vector):
        param.data.add_(task_vector_coefficient * tv.to(param.device))
    return model

def prepare_task_vectors(model: FSDP, model_tv: FSDP, model_storage):
    task_vectors = []
    for i, (param, param_tv) in enumerate(zip(fsdp_v1_model_params(model), fsdp_v1_model_params(model_tv))):
        task_vectors.append(param_tv.detach().cpu() - param.detach().cpu())
    del model_tv
    torch.cuda.empty_cache()
    return task_vectors
def _filter_dpo_inputs(
    inputs: Dict[str, torch.Tensor], chosen: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Filter inputs for Direct Preference Optimization (DPO) based on whether they are chosen or rejected.

    This function takes a dictionary of input tensors and filters them based on whether they
    are for the chosen or rejected option in a DPO setup.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.
        chosen (bool, optional): A flag indicating whether to filter for chosen or rejected inputs.
                                 Defaults to False (i.e., rejected inputs).

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the relevant input tensors.
    """
    prefix = "chosen_" if chosen else "rejected_"
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filter the input dictionary to keep only specific keys.

    This function takes a dictionary of input tensors and returns a new dictionary
    containing only the keys 'input_ids', 'attention_mask', and 'labels' if they exist
    in the original dictionary.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the specified keys.
    """
    return {
        k: v
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }
    
def log_p_loss(
    logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    Compute the log probability loss for a language model.

    This function calculates the cross-entropy loss between the predicted logits
    and the true labels, typically used in language modeling tasks.

    Args:
        logits (torch.Tensor): The predicted logits from the model, typically of shape
                               (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): The true labels, typically of shape
                               (batch_size, sequence_length).
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def obj_standard_max_next_token(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    accelerator: Optional[Accelerator] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Compute the standard maximum next token objective.

    This function calculates the log probability loss for the next token prediction
    using the given model and inputs. It supports both standard inputs and
    Direct Preference Optimization (DPO) inputs.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        inputs (Dict[str, torch.Tensor]): The input tensors for the model.
        accelerator (Optional[Accelerator]): The Accelerator object for distributed training. Defaults to None.
        chosen (bool): Flag to indicate whether to use chosen or rejected inputs for DPO. Defaults to False.

    Returns:
        torch.Tensor: The computed log probability loss.
    """
    outputs = model(
        **_filter_inputs(_filter_dpo_inputs(inputs, chosen)), output_hidden_states=False
    )
    return log_p_loss(
        outputs.logits,
        _filter_dpo_inputs(inputs, chosen).get("labels"),
        model.vocab_size,
    )


def delete_optimizer(optim):
    # go through all states and delete the param groups
    for state in optim.state.values():
        state.clear()
    optim.param_groups = []
    del optim
    torch.cuda.empty_cache()

def get_next_batch(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def next_n_batches(iterator, dataloader, n):
    batches = []
    for _ in range(n):
        batch, iterator = get_next_batch(iterator, dataloader)
        batches.append(batch)
    return batches, iterator

def get_distributed_random_number(accelerator: Accelerator):
    random_number = torch.rand(1).to(accelerator.device)
    dist.broadcast(random_number, src=0)
    accelerator.wait_for_everyone()
    return random_number.item()

def distributed_sample_task(adversaries):
    # generate shared random number across all GPUs via broadcasting:
    # e.g., {task1: 0.33, task2: 0.66, task3: 0.01} etc
    task_probs = {
        adv.split(":")[0]: float(adv.split(":")[1]) for adv in adversaries.split(",")
    }
    task_type = random.choices(
        list(task_probs.keys()), weights=list(task_probs.values()), k=1
    )[0]
    dist.barrier()
    task_type = accelerate_utils.broadcast_object_list([task_type], 0)[0]
    return task_type


def distributed_sample_adversary_lr(adversary_lr_samples, accelerator):
    dist.barrier()
    rand_num = get_distributed_random_number(accelerator)
    adversary_lr = adversary_lr_samples[
        math.floor(rand_num * len(adversary_lr_samples))
    ]
    return adversary_lr