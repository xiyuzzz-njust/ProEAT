import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Any, Union, List, Optional
from gcg_attack import GCGAttack
from contextlib import nullcontext

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, reference_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcg_attack = GCGAttack(self.model, self.tokenizer)
        self.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"


        self.reference_model = reference_model
        self._peft_has_been_casted_to_bf16 = hasattr(self.model, "is_loaded_in_bf16") and self.model.is_loaded_in_bf16
 
        self.momentum = 0.9  #
        self.normal_loss_ma = None 
        # self.original_loss_ma = None
        self.adv_loss_ma = None
        

        self.min_weight = 0
        self.max_weight = 1.0
        
        self.ref_alpha = 0.1
        print(self.args.gcg_step, self.args.pgd_iterations)


    def update_moving_average(self, old_ma, new_value):
        if old_ma is None:
            return new_value
        return self.momentum * old_ma + (1 - self.momentum) * new_value

    def compute_dynamic_weights(self, normal_ma, adv_ma):
        # Calculate weight based on moving average
        total = normal_ma + adv_ma
        weights = torch.tensor([
            normal_ma/total,
            # original_ma/total, 
            adv_ma/total
        ], device=normal_ma.device)
        
        # Weight restriction and normalization
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        
        return weights
    
    def compute_reference_log_probs(self, inputs: Dict) -> torch.Tensor:   # 2025-01-08
        compute_ref_context_manager = ( torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext )
        
        with torch.no_grad(), compute_ref_context_manager():
            if self.reference_model is None:
                with self.compute_loss_context_manager():
                    outputs = self.model(**inputs)
            else:
                outputs = self.reference_model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    use_cache=False,
                )
            
            logits = outputs.logits
            if logits.shape[1] != inputs["labels"].shape[1]:
                min_len = min(logits.shape[1], inputs["labels"].shape[1])
                logits = logits[:, :min_len, :]
                labels = inputs["labels"][:, :min_len]
            else:
                labels = inputs["labels"]
                
            # print(f"logits shape: {logits.shape}")
            # print(f"labels shape: {labels.shape}")
            
            log_probs = self.get_batch_logps(logits, labels)
            
            return log_probs

    def train_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.empty_cache()
        model.train()
        normal_inputs = self._prepare_inputs(inputs["normal"])
        adv_inputs = self._prepare_inputs(inputs["adversarial"])
        
        with self.compute_loss_context_manager():
            normal_loss = self.compute_loss(model, normal_inputs)

            normal_ref_logps = self.compute_reference_log_probs(normal_inputs)

        init_perturbed_inputs = {
            "input_ids": adv_inputs["input_ids"].clone(), 
            "labels": adv_inputs["labels"].clone()
        }

        control_tokens = self.tokenizer.encode(self.control_init, add_special_tokens=False)     
        control_tokens = torch.tensor(control_tokens, device=adv_inputs["input_ids"].device)

        # conv = conversation_lib.default_conversation.copy()
        # assistant_start_str = conv.sep + "Assistant:"
        assistant_token = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        assistant_token = torch.tensor(assistant_token, device=adv_inputs["input_ids"].device)

        # Find the end position of human input in each sequence and insert control tokens
        batch_size, seq_length = adv_inputs["input_ids"].shape
        control_length = len(control_tokens)

        new_seq_length = seq_length + control_length
        new_input_ids = torch.full((batch_size, new_seq_length), 
                                self.tokenizer.pad_token_id, 
                                device=adv_inputs["input_ids"].device)
        new_labels = torch.full((batch_size, new_seq_length), 
                            IGNORE_INDEX, 
                            device=adv_inputs["labels"].device)
        
        control_slices = []
        for i in range(batch_size):
            assistant_pos = None
            for j in range(seq_length - len(assistant_token) + 1):
                if torch.all(adv_inputs["input_ids"][i, j:j+len(assistant_token)] == assistant_token):
                    assistant_pos = j
                    break

            human_end = assistant_pos 
            control_slices.append(slice(human_end, human_end + control_length))

            last_non_pad = adv_inputs["attention_mask"][i].nonzero()[-1].item() + 1
            assistant_start = (adv_inputs["labels"][i] != IGNORE_INDEX).nonzero(as_tuple=True)[0][0]
            
            new_input_ids[i, :human_end] = adv_inputs["input_ids"][i, :human_end]  
            new_input_ids[i, human_end:human_end + control_length] = control_tokens  
            new_input_ids[i, human_end + control_length:control_length + last_non_pad] = adv_inputs["input_ids"][i, human_end:last_non_pad]

            
            new_labels[i, :assistant_start + control_length] = IGNORE_INDEX
            new_labels[i, assistant_start + control_length:control_length + last_non_pad] = adv_inputs["labels"][i, assistant_start:last_non_pad]
        
        init_perturbed_inputs = {
            "input_ids": new_input_ids,
            "attention_mask": new_input_ids.ne(self.tokenizer.pad_token_id),
            "labels": new_labels
        }

        adv_gcg_inputs = self.gcg_attack.run_attack(
            input_ids=init_perturbed_inputs["input_ids"],
            attention_mask=init_perturbed_inputs["attention_mask"],
            labels=init_perturbed_inputs["labels"],
            pgd_input_ids=adv_inputs["input_ids"],
            pgd_attention_mask=adv_inputs["attention_mask"],
            pgd_labels=adv_inputs["labels"],
            images=adv_inputs["images"],
            targets=adv_inputs["targets"],
            control_slices=control_slices,
            n_steps=self.args.gcg_step,
            pgd_iterations=self.args.pgd_iterations
        )


        with self.compute_loss_context_manager():
            adv_loss = self.compute_loss(model, adv_gcg_inputs)
            adv_ref_logps = self.compute_reference_log_probs(adv_gcg_inputs)

        self.normal_loss_ma = self.update_moving_average(self.normal_loss_ma, normal_loss.detach())
        self.adv_loss_ma = self.update_moving_average(self.adv_loss_ma, adv_loss.detach())

        weights = self.compute_dynamic_weights(
            self.normal_loss_ma,
            self.adv_loss_ma
        )

        base_loss = (weights[0] * normal_loss.mean() + weights[1] * adv_loss.mean())
        ref_loss = self.ref_alpha * (
            weights[0] * (normal_loss - normal_ref_logps).mean() +
            weights[1] * (adv_loss - adv_ref_logps).mean()
        )
        
        loss = base_loss + ref_loss
        print(f"Dynamic weights: normal={weights[0]:.4f}, adv={weights[1]:.4f}")
        print(f"Normal loss: {normal_loss:.4f}, Adv loss: {adv_loss:.4f}")

        # loss = 0.5*normal_loss + 0.5*adv_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    @staticmethod   # 2025-01-08
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        upper_cutoff: float = -10.0,
        lower_cutoff: float = 0.0,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # Zero out the loss for tokens where log(P(label)) is less than the threshold
        below_threshold = per_token_logps < lower_cutoff
        per_token_logps[below_threshold] = 0

        # Zero out the loss for tokens where log(P(label)) is greater than the threshold
        above_threshold = per_token_logps > upper_cutoff
        per_token_logps[above_threshold] = 0

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        return self.train_step(model, inputs) 
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    
                ]
                #print(optimizer_grouped_parameters)
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
