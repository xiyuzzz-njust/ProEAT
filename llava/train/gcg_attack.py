import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class GCGAttack:
    def __init__(self, model, tokenizer):
        self.model = model 
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        

    def target_loss(self, logits: torch.Tensor, target_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size = logits.shape[0]
        losses = []
        for i in range(batch_size):
            curr_target = target_ids[i]
            target_len = len(curr_target)
            available_len = logits[i].shape[0] - start_pos
            
            if available_len < target_len:
                curr_target = curr_target[:available_len]
                target_len = available_len
                
            curr_logits = logits[i, start_pos:start_pos+target_len, :]
            loss = F.cross_entropy(curr_logits, curr_target, reduction='mean')
            losses.append(loss)
        return torch.stack(losses)

    def get_grad(self, model: nn.Module, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, target_ids: torch.Tensor,
                control_slice: slice, start_pos: int) -> torch.Tensor:
        # Get embedding layer weight
        embed_weights = model.model.embed_tokens.weight
        input_ids[input_ids == IMAGE_TOKEN_INDEX] = self.tokenizer.pad_token_id

        one_hot = torch.zeros(
            input_ids[control_slice].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[control_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype)
        ).to(dtype=embed_weights.dtype, device=embed_weights.device)
        one_hot.requires_grad_(True)

        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # Get the embeddings of the complete sequence
        embeds = model.model.embed_tokens(input_ids.unsqueeze(0)).detach()

        full_embeds = torch.cat([
            embeds[:,:control_slice.start,:],
            input_embeds,
            embeds[:,control_slice.stop:,:]
        ], dim=1)


        full_embeds.requires_grad_(True)

        outputs = model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask.unsqueeze(0),
        )
        logits = outputs.logits

        loss = self.target_loss(logits, target_ids.unsqueeze(0), start_pos)
        model.zero_grad()
        loss.backward()
        
        return one_hot.grad.clone()
        
    def pgd(
        self,
        model, 
        data_clean,  # Clean input data (undisturbed image)
        data_text,
        attention_mask,
        start_positions,
        targets,
        norm, 
        eps,  
        iterations,  
        stepsize,  
        output_normalize, 
        perturbation=None,  # initial disturbance (if not given, it is initialized to zero tensor)
        mode='min',  # optimization mode‘ Min 'means to minimize the loss‘ Max 'means maximum loss
        momentum=0.9,
        verbose=True, 
    ):
        """
        Minimize or maximize given loss
        """
        batch_size = data_clean.shape[0]
        device = data_clean.device
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)

        eps_tensor = torch.full((batch_size, 3, 1, 1), eps / 255, device=device)
        dis_tensor = torch.full((batch_size, 3, 1, 1), 0, device=device)
        
        eps = normalize(eps_tensor, image_mean, image_std)
        dis = normalize(dis_tensor, image_mean, image_std)
        eps = eps - dis
        #stepsize = eps / 4   
        stepsize_tensor = torch.full((batch_size, 3, 1, 1), stepsize / 255, device=device)
        stepsize = normalize(stepsize_tensor, image_mean, image_std)
        stepsize = stepsize - dis

        perturbation = normalize(torch.zeros_like(data_clean, device=device).uniform_(-4 / 255, 4 / 255), image_mean, image_std)  - dis
        perturbation.requires_grad_(True)

        velocity = torch.zeros_like(data_clean, device=device)


        for i in range(iterations):
            perturbation.requires_grad = True
            with torch.enable_grad():
                input_image = data_clean + perturbation

                outputs = model(
                    input_ids=data_text,
                    images=input_image,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                loss = 0
                for b in range(batch_size):
                    loss += self.target_loss(logits[b:b+1], targets[b:b+1], start_positions[b])
                loss = loss / batch_size

                if verbose:
                    print(f'[{i}] {loss.item():.5f}')
                model.zero_grad()
                if perturbation.grad is not None:
                    perturbation.grad.zero_()
                loss.backward()

                gradient = perturbation.grad.clone()
                # print(f'[{i}], loss: {loss.item():.5f}, max_grad: {gradient.abs().max().item()}')

            with torch.no_grad():

                if gradient.isnan().any():  #
                    print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                    gradient[gradient.isnan()] = 0.
                # normalize
                gradient = normalize_grad(gradient, p=norm)
                # momentum
                velocity = momentum * velocity + gradient
                velocity = normalize_grad(velocity, p=norm)
                # update
                if mode == 'min':
                    perturbation = perturbation - stepsize * velocity
                elif mode == 'max':
                    perturbation = perturbation + stepsize * velocity

                else:
                    raise ValueError(f'Unknown mode: {mode}')

                perturbation = project_perturbation(perturbation, eps, norm)

                denormalized_image = denormalize(data_clean + perturbation, image_mean, image_std)

                denormalized_image = torch.clamp(denormalized_image, 0, 1)
                
                perturbation = normalize(denormalized_image, image_mean, image_std) - data_clean

                model.zero_grad()

        return data_clean + perturbation.detach()


    def run_attack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pgd_input_ids: torch.Tensor,
        pgd_attention_mask: torch.Tensor,
        pgd_labels: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
        control_slices: List[slice],
        n_steps: int = 500,
        batch_size: int = 512,
        topk: int = 256,
        temp: float = 1.0,
        pgd_iterations: int = 10
    ) -> Dict[str, torch.Tensor]:

        best_losses = torch.full((input_ids.shape[0],), float('inf'))
        best_input_ids = input_ids.clone()

        start_positions = [(labels[i] != -100).nonzero(as_tuple=True)[0][0].item() for i in range(labels.shape[0])]
        
        for step in range(n_steps):
            for batch_idx in range(input_ids.shape[0]):

                grad = self.get_grad(
                    self.model,
                    best_input_ids[batch_idx].clone(),
                    attention_mask[batch_idx].clone(), 
                    targets[batch_idx],
                    control_slices[batch_idx],
                    start_positions[batch_idx],
                )

                control_len = control_slices[batch_idx].stop - control_slices[batch_idx].start

                # Obtain the top-k candidate token of each location according to the gradient
                control_toks = best_input_ids[batch_idx, control_slices[batch_idx]]
                top_indices = (-grad).topk(topk, dim=1).indices

                
                num_candidates = min(batch_size, topk)
                candidates = best_input_ids[batch_idx].clone().repeat(num_candidates, 1)
                
                # Random sampling token for each location
                for pos in range(control_len):
                    probs = F.softmax(temp * torch.ones(topk), dim=0) 
                    new_tokens = top_indices[pos][torch.multinomial(probs, num_candidates)]
                    candidates[:, control_slices[batch_idx].start + pos] = new_tokens
                
                candidates_copy = candidates.clone()
                candidates_copy[candidates_copy == IMAGE_TOKEN_INDEX] = self.tokenizer.pad_token_id

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=candidates_copy,
                        attention_mask=attention_mask[batch_idx].repeat(num_candidates, 1)
                    )
                    
                    # Predict target from the unified start_pos position
                    cand_loss = self.target_loss(outputs.logits, targets[batch_idx].repeat(num_candidates, 1), start_positions[batch_idx])
                    
                # Select the best candidate
                best_cand_idx = cand_loss.argmin()
                if cand_loss[best_cand_idx] < best_losses[batch_idx]:
                    # print(f"Step {step}, batch {batch_idx}, loss: {cand_loss[best_cand_idx].item()}")
                    best_losses[batch_idx] = cand_loss[best_cand_idx].item()
                    best_input_ids[batch_idx] = candidates[best_cand_idx]

                del grad, candidates, outputs
                torch.cuda.empty_cache()
            

        images_adv = self.pgd(
            model=self.model,
            data_clean=images,
            data_text=pgd_input_ids,
            attention_mask=pgd_attention_mask,
            start_positions=start_positions,
            targets=targets,
            norm='linf',
            eps=8,
            iterations=pgd_iterations,
            stepsize=2,
            output_normalize=False,
            perturbation=torch.zeros_like(images).uniform_(-4, 4).requires_grad_(True),
            mode='min',
            verbose=True,
        )

        if n_steps == 0 and pgd_iterations != 0:
            return {
                "input_ids": pgd_input_ids,
                "attention_mask": pgd_attention_mask,
                "labels": pgd_labels,
                "images": images_adv
            }
        elif pgd_iterations == 0 and n_steps != 0:
            return {
                "input_ids": best_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "images": images,
            }
        elif pgd_iterations == 0 and n_steps == 0:
            return {
                "input_ids": pgd_input_ids,
                "attention_mask": pgd_attention_mask,
                "labels": pgd_labels,
                "images": images,
            }
        else:
            return {
                "input_ids": best_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "images": images_adv
            }
        

def normalize(image_tensor, mean, std):
    return (image_tensor - mean) / std

def denormalize(image_tensor, mean, std):
    return image_tensor * std + mean

def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')
    
def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf']:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)