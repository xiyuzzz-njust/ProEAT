from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import copy
import torch.nn.functional as F
import os
from tqdm import tqdm

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
    
class CustomImageDataset(Dataset):
    def __init__(self, folder_path, image_processor, model_config):
        # 获取文件夹中所有图片文件的路径
        self.image_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))  # 根据需要添加其他图片格式
        ]
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图片路径
        image_path = self.image_paths[idx]

        # 加载并转换为 RGB 格式
        image = Image.open(image_path).convert('RGB')

        # 处理图片，获取 image tensor
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        #image_tensor = image_tensor

        # 返回处理后的图片张量
        return image_tensor

class ComputeLossWrapper:
    def __init__(self, embedding_orig):
        self.embedding_orig = embedding_orig

    def __call__(self, embedding):
        return compute_loss(
            embedding=embedding,
            embedding_orig=self.embedding_orig
            )

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
      
def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch


def compute_loss(embedding, embedding_orig, reduction='mean'):
    loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)

    return loss


def normalize(image_tensor, mean, std):
    return (image_tensor - mean) / std

def denormalize(image_tensor, mean, std):
    return image_tensor * std + mean

def pgd(
    model,  # 前向传播函数，接受输入数据并返回模型输出
    loss_fn,  # 损失函数，用于计算模型输出与目标之间的误差
    data_clean,  # 干净的输入数据（未经扰动的图像）
    norm,  # 范数类型（比如 L2 或 L∞），用于衡量扰动的大小
    eps,  # 最大扰动大小
    iterations,  # PGD算法的迭代次数
    stepsize,  # 每次迭代更新扰动的步长
    output_normalize,  # 是否在前向传播时对输出进行归一化
    perturbation=None,  # 初始扰动（如果没有给定，则初始化为零张量）
    mode='min',  # 优化模式，‘min’表示最小化损失，‘max’表示最大化损失
    momentum=0.9,  # 动量，用于控制扰动更新时的惯性
    verbose=True,  # 是否输出每次迭代的损失信息
    
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space

    #assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    # if perturbation is None:
    #     perturbation = torch.zeros_like(data_clean, requires_grad=True)
    
    batch_size = data_clean.shape[0]
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)

    eps_tensor = torch.full((batch_size, 3, 1, 1), eps / 255)
    dis_tensor = torch.full((batch_size, 3, 1, 1), 0)
    
    eps = normalize(eps_tensor, image_mean, image_std)
    dis = normalize(dis_tensor, image_mean, image_std)
    eps = eps - dis


    
    stepsize_tensor = torch.full((batch_size, 3, 1, 1), stepsize / 255)
    stepsize = normalize(stepsize_tensor, image_mean, image_std)
    stepsize = stepsize - dis

    perturbation = normalize(torch.zeros_like(data_clean).uniform_(-4 / 255, 4 / 255), image_mean, image_std)  - dis
    perturbation.requires_grad_(True)

    velocity = torch.zeros_like(data_clean)


    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            #perturbation = perturbation.to(torch.float16)

            image_features = model.get_model().get_vision_tower()(data_clean + perturbation)

            out = model.get_model().mm_projector(image_features)  
            loss = loss_fn(out)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            gradient = gradient

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
            # project
            #print(f"perturbation before {perturbation}")
            perturbation = project_perturbation(perturbation, eps, norm)

            denormalized_image = denormalize(data_clean + perturbation, image_mean, image_std)


            denormalized_image = torch.clamp(denormalized_image, 0, 1)
            
            # 重新标准化到网络的输入空间
            perturbation = normalize(denormalized_image, image_mean, image_std) - data_clean

            # assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
            #     data_clean + perturbation
            # ) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return data_clean + perturbation.detach()

if __name__ == "__main__":
    device = "cuda"
    torch.set_default_device(device)
    model_path = "/mnt/beegfs/home/zengxiyu24/RobustVLM/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    print("initial model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device, torch_dtype=torch.float32
    )

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    clean_data = CustomImageDataset(
        "/mnt/beegfs/home/zengxiyu24/LLaVA/playground/data/imagenet/my_test_sampled",
        image_processor,
        model.config
    )

    data_loader = torch.utils.data.DataLoader(
        clean_data, batch_size=2, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.get_model().mm_projector.parameters(), lr=1e-5, weight_decay=1e-4
    )

    vision_tower = model.get_model().get_vision_tower()
    origin_projector = copy.deepcopy(model.get_model().mm_projector)
    projector = model.get_model().mm_projector

    total_steps = 20000
    steps_per_epoch = len(data_loader)
    num_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch

    step_total = 0
    progress_bar = tqdm(total=total_steps, desc='Total Progress', position=0)

    for epoch in range(num_epochs):
        epoch_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                        position=1, leave=False)
        
        for data in epoch_bar:
            if step_total >= total_steps:
                break
                
            data = data.to(device)

            with torch.no_grad():
                feature_orig = vision_tower(data)
                embedding_orig = origin_projector(feature_orig)

            loss_inner_wrapper = ComputeLossWrapper(embedding_orig)
            
            model.eval()
            data_adv = pgd(
                model=model,
                loss_fn=loss_inner_wrapper,
                data_clean=data,
                norm='linf',
                eps=8,
                iterations=50,
                stepsize=2,
                output_normalize=False,
                perturbation=torch.zeros_like(data).uniform_(-4, 4).requires_grad_(True),
                mode='max',
                verbose=False,  # 关闭PGD的详细输出，避免干扰进度条
            )

            del loss_inner_wrapper

            model.train()
            image_features = vision_tower(data_adv)
            embedding_adv = model.get_model().mm_projector(image_features) 
            del data, data_adv

            loss = compute_loss(embedding_adv, embedding_orig) 

            print(f"loss: {loss.item()}")
            
            # 更新进度条信息
            epoch_bar.set_postfix({'loss': f'{loss.item():.5f}'})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step_total += 1
            progress_bar.update(1)
            
            if step_total % 1000 == 0:
                save_path = f"./mm_projector_8_50/mm_projector_{step_total}.bin"
                torch.save(model.get_model().mm_projector.state_dict(), save_path)
                tqdm.write(f"Checkpoint saved: {save_path}")

        epoch_bar.close()

    progress_bar.close()
    
    # Save the final model
    final_save_path = "./mm_projector_8_50/mm_projector.bin"
    torch.save(model.get_model().mm_projector.state_dict(), final_save_path)
    print(f"Final model saved: {final_save_path}")