import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import ConvGN
from diffusers import UniPCMultistepScheduler, UNet2DConditionModel, PNDMScheduler,  AutoPipelineForText2Image, StableDiffusionPipeline




class DiffSeg(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        decode_intermediate_input=True,
        hidden_dim=256,
        shortcut_dims=[24, 32, 96, 1280],
        align_corners=True,
        
        model =  "UNet",
        schedule ="DDPM"  ,
        conv_hidden_dim = 256,
        timestep =  50,
        condition_dim = 1280,
        generate_seed = 0,
        guidance_scale = 0.5,
        # predict_one_hot = True
        
    ):

        super().__init__()
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        self.conv_in = ConvGN(in_dim, hidden_dim, 1)

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)

        self.obj_num = out_dim

        self.condition_Conv = Condition_ConvReducer(  hidden_dim // 2 , output_dim= conv_hidden_dim)

        self.condition_MLP = Condition_MLP( input_dim= conv_hidden_dim , num_features= condition_dim )

        self.uncond_embed = nn.Parameter(torch.randn(1, 1, condition_dim))
        
        self.generator = torch.manual_seed(generate_seed)


        # ! 这两个config
        self.scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.model = UNet2DConditionModel( sample_size= 117,in_channels= self.obj_num , out_channels= self.obj_num)
        # self.model = UNet2DConditionModel.from_pretrained( "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True,in_channels = self.obj_num)

        self.scheduler.set_timesteps(timestep)
        self.guidance_scale = guidance_scale
        # self.predict_one_hot = predict_one_hot


        self._init_weight()

    
    def get_condition(self, inputs, shortcuts):


        device = inputs[-1].device

        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

        # x.shape = [2, 1024, 30, 30]
        x = F.relu_(self.conv_in(x))
        # x.shape = [2, 256, 30, 30]
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-2]) + x))
        # x.shape = [2, 256, 30, 30]

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        # x.shape = [2, 256, 59, 59]
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-3]) + x))
        # x.shape = [2, 128, 59, 59]

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        # x.shape = [2, 128, 117, 117]
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-4]) + x))

        _, _ , self.output_h , self.output_w  = x.shape

        x = self.condition_Conv(x)
        bs, fd, h, w = x.shape
        x = torch.permute(x, (0,2,3,1))
        x = x.reshape(bs * h * w, fd)
        
        x = self.condition_MLP(x)

        x = x.reshape(bs, h * w, -1)


        return x

    def inference(self,inputs, shortcuts):

        cond_embed = self.get_condition(inputs, shortcuts)
        batch_size , patch_size , _ = cond_embed.shape
        device = cond_embed.device

        generator = torch.Generator(device=device).manual_seed(self.generator.initial_seed())
        

        uncond_embed = self.uncond_embed.repeat(batch_size , patch_size , 1).to(device)

        memory_embed = torch.cat([uncond_embed, cond_embed])

        latents = torch.randn(
            (batch_size, self.model.config.in_channels, self.output_h, self.output_w),
            generator=generator,
            device= device,
        )

        latents = latents * self.scheduler.init_noise_sigma

        for t in (self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.model(latent_model_input, t, encoder_hidden_states= memory_embed  ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            
        return latents
    
    def Diffusion_train(self,inputs, shortcuts, gt_mask):
        cond_embed = self.get_condition(inputs, shortcuts)
        batch_size , patch_size , _ = cond_embed.shape
        device = cond_embed.device
        uncond_embed = self.uncond_embed.repeat(batch_size , patch_size , 1).to(device)
        # uncond_embed = self.uncond_embed.repeat(batch_size , 1)

        generator = torch.Generator(device=device).manual_seed(self.generator.initial_seed())

        memory_embed = cond_embed

        noise  = torch.randn(gt_mask.shape, device=gt_mask.device)

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device= gt_mask.device,
            dtype=torch.int64
        )
        timesteps = timesteps.long()


        
        noisy_images = self.scheduler.add_noise(gt_mask, noise, timesteps)


        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if False and self.conditioning_dropout_prob is not None:
            random_p = torch.rand(batch_size, device= device, generator=generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
            # Final text conditioning.
            # null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
            encoder_hidden_states = torch.where(prompt_mask,  uncond_embed , memory_embed)
        else:
            encoder_hidden_states = memory_embed
        
        noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states ).sample
        loss = F.mse_loss(noise_pred, noise)


        return loss


    def forward(self, inputs, shortcuts):


        device = inputs[-1].device

 

        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

        # x.shape = [2, 1024, 30, 30]
        x = F.relu_(self.conv_in(x))
        # x.shape = [2, 256, 30, 30]
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-2]) + x))
        # x.shape = [2, 256, 30, 30]

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        # x.shape = [2, 256, 59, 59]
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-3]) + x))
        # x.shape = [2, 128, 59, 59]

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        # x.shape = [2, 128, 117, 117]
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-4]) + x))
        # x.shape = [2, 128, 117, 117]

        x = self.conv_out(x)

        n,c ,h ,w = x.shape

        noise = torch.randn((n,c,h,w), device= device)

        # x.shape = [2, 11, 117, 117]

        return x
    


    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Calculate the total memory consumed by the model's parameters
        total_memory = 0
        for name, param in self.named_parameters():
            param_memory = param.numel() * param.element_size()
            total_memory += param_memory
            # print(f"Name: {name}, Shape: {param.shape}, Memory: {param_memory} bytes")

        print(f"DiffSeg Total memory: {total_memory / (1024 ** 2):.2f} MB")


class Condition_ConvReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Condition_ConvReducer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, output_dim * 2, kernel_size=3, stride=2, padding=1),  # Output: (batch, 64, 59, 59)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 64, 30, 30)
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, stride=2, padding=1),  # Output: (batch, 32, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: (batch, 32, 7, 7)
        )

    def forward(self, x):
        return self.features(x)
    


class Condition_MLP(nn.Module):
    def __init__(self, input_dim, num_features):
        super(Condition_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_features)
        )

    def forward(self, x):
        x = self.flatten(x)

        return self.linear_layers(x)