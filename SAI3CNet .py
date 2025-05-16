import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
import types

class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, 
                                 kernel_size=(3, 15), padding=(1, 7), stride=(2, 2), bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        return x
    

class ResBlock(nn.Module):
    def __init__(self, out_ch, downsample):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=(3, 9),stride=(self.stride, self.stride),padding=(1, 4),bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=(3, 9) ,padding=(1, 4),bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(2, 2))
            self.idfunc_1 = nn.Conv2d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x):
        identity = x
        if x.size(2) % 2 != 0:
            identity = F.pad(identity, (1, 0, 0, 0))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
            
        x = x + identity
        return x



class SpAttn_LRE(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpAttn_LRE, self).__init__()
        
    
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, 
                                padding=kernel_size // 2, bias=False) 

    def forward(self, x):
 
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (batch, 1, 128, 4, 8)
    
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, 128, 4, 8)

        attention_input = torch.cat([max_pool, avg_pool], dim=1)  # (batch, 2, 128, 4, 8)

        attention = torch.sigmoid(self.conv3d(attention_input))  # (batch, 1, 128, 4, 8)

        out = x + x * attention  # (batch, 2, 128, 4, 8)

        return out

class LeadRelationShipEncoder(nn.Module):
    def __init__(self, lead, out_ch):
        super(LeadRelationShipEncoder, self).__init__()
        self.conv2_1 = nn.Conv3d(in_channels=2, out_channels=1,
                                 kernel_size=1,
                                 bias=False)

        self.pool2_1 = nn.AvgPool2d(kernel_size=(4, 8), stride=(4, 4))
        
        self.conv2_2 = nn.Conv3d(in_channels=4, out_channels=1,
                                 kernel_size=1,
                                 bias=False)

        self.pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool2 = nn.AdaptiveAvgPool1d(output_size=1)

        self.spatial_attention = SpAttn_LRE()

    def forward(self, x):
        tensor1 = [x[:, :, :, i*8:(i+1)*8] for i in range(4)] 

        x1 = torch.stack((tensor1[0], tensor1[1]), dim=1)  # B, 2, C, H, W
        x1 = self.spatial_attention(x1)  # B, 2, C, H, W
        x1 = self.conv2_1(x1)  # B, 1, C, H, W
        x1 = x1.squeeze(1)  # B, C, H, W
        x1 = self.pool2_1(x1)  # B, C, H', W'
        

        x2 = torch.stack((tensor1[2], tensor1[3]), dim=1)
        x2 = self.spatial_attention(x2)  # B, 2, C, H, W
        x2 = self.conv2_1(x2)
        x2 = x2.squeeze(1)
        x2 = self.pool2_1(x2)
        
        x3 = torch.stack((tensor1[0], tensor1[1], tensor1[2], tensor1[3]), dim=1)
        x3 = self.spatial_attention(x3)
        x3 = self.conv2_2(x3)
        x3 = x3.squeeze(1)
        x3 = self.pool2_1(x3)

        x1 = x1.flatten(1)  # B, C*H'*W'
        x2 = x2.flatten(1)
        x3 = x3.flatten(1)


        x = torch.cat([x1, x2, x3], dim=1)  # B, 3*C*H'*W'
        return x
    

class SpAttn(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpAttn, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size), 
                             padding=(kernel_size//2, kernel_size//2, kernel_size//2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 1, 4, 128, 256)

        x = torch.chunk(x, 4, dim=-1)
        x = torch.stack(x, dim=2)
        
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (batch, 1, 4, 128, 256)
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (batch, 1, 4, 128, 256)

        pools = torch.cat([max_pool, avg_pool], dim=1)   # (batch, 2, 4, 128, 256)
        
        attention = self.sigmoid(self.conv(pools))       # (batch, 1, 4, 128, 256)
        x = x + x * attention
        x = torch.cat(torch.unbind(x, dim=2), dim=-1)
        return x
    
class SAI3CNet (nn.Module):
    def __init__(self, nOUT, in_ch = 1, out_ch = 128, lead = 20):
        super(SAI3CNet , self).__init__()
        self.inconv = INCONV(in_ch=in_ch, out_ch=out_ch)

        self.rb_0 = nn.Sequential(
            ResBlock(out_ch=out_ch, downsample=True),
            ResBlock(out_ch=out_ch, downsample=False)
        )
        self.rb_1 = nn.Sequential(
            ResBlock(out_ch=out_ch, downsample=True),
            ResBlock(out_ch=out_ch, downsample=False)
        )
        self.rb_2 = nn.Sequential(
            ResBlock(out_ch=out_ch, downsample=True),
            ResBlock(out_ch=out_ch, downsample=False)
        )
        self.rb_3 = nn.Sequential(
            ResBlock(out_ch=out_ch, downsample=True),
            ResBlock(out_ch=out_ch, downsample=False)
        )

        self.attention_0 = SpAttn()
        self.attention_1 = SpAttn()
        self.attention_2 = SpAttn()
        self.attention_3 = SpAttn()
        self.attention_4 = SpAttn()


        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.lre = LeadRelationShipEncoder(lead=lead, out_ch=out_ch)

        self.mlp = nn.Sequential(
            nn.LayerNorm(out_ch*3),
            nn.Linear(out_ch*3, out_ch*3),
            nn.ReLU(),
            nn.Linear(out_ch*3, out_ch*3)
        )
        self.fc = nn.Linear(out_ch*3, nOUT)

    def forward(self, x):

        x = self.attention_0(x) # B, 1, 128, 1024
        x = self.inconv(x) # B, 1, 64, 512

        #x - self.attention_1(x) # B, 1, 64, 512
        x = self.rb_0(x) # B, 1, 32, 256

        x = self.attention_2(x) # B, 1, 32, 256
        x = self.rb_1(x) # B, 1, 16, 128


        x = self.attention_3(x) # B, 1, 16, 128
        x = self.rb_2(x) # B, 1, 8, 64

        #x = self.attention_4(x) # B, 1, 8, 64
        x = self.rb_3(x) # B, 1, 4, 32
        x= self.lre(x)
        
        embeddings = self.mlp(x)
        x = self.fc(embeddings)

        return x, embeddings
    


# def modify_model(model, device, nOUT):
#     model.lc = nn.Identity()
#     model.mlp = nn.Identity()
#     model.fc = nn.Identity()



#     model.pool = nn.AdaptiveAvgPool2d(output_size=1).to(device)
#     model.fc_pool = nn.Linear(128, nOUT).to(device)

#     def new_forward(self, x):
#         print(x.shape)
#         x = self.inconv(x)
#         print(x.shape)

#         x = self.rb_0(x)
#         x = self.rb_1(x)
#         x = self.rb_2(x)
#         x = self.rb_3(x)
#         x = self.pool(x)
#         embedding = x.squeeze()
#         x = self.fc_pool(embedding)
#         return x, embedding
    
#     model.forward = types.MethodType(new_forward, model)
    
#     return model
        



if __name__ == "__main__":
    input = torch.rand(4, 1, 128, 1024).to("cuda")
    model = SAI3CNet (nOUT=6).to("cuda")

    #modified_model = modify_model(model, device="mps", nOUT=6)

    #output, embedding = modified_model(input)


    out, embedding = model(input)
    print(out.shape, embedding.shape)