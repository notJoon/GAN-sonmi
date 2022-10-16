import torch.nn as nn
import torch 

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, blocks=5, res_scale=0.2) -> None:
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True) -> nn.Sequential:
            layers = [nn.Conv2d(in_features, filters, kernel_size=3, stride=1, padding=1, bias=True)]

            if non_linearity:
                layers += [nn.LeakyReLU()]
            
            return nn.Sequential(*layers)

        self.b1 = block(in_features= 1 * filters) 
        self.b2 = block(in_features= 2 * filters)
        self.b3 = block(in_features= 3 * filters)
        self.b4 = block(in_features= 4 * filters)
        self.b5 = block(in_features= 5 * filters, non_linearity=False)
        
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x) -> torch.Tensor:
        inputs = x
        for block in self.blocks():
            x = block(x)
            inputs = torch.cat([inputs, x], dim=1)
        
        return x.mul(self.res_scale) + x 


class ResidualInResidualDensBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2, dense_layers=4) -> None:
        super(ResidualInResidualDensBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_block = nn.Sequential(
            DenseResidualBlock(filters), 
            DenseResidualBlock(filters), 
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
        )

    def forward(self, x) -> torch.Tensor:
        return self.dense_block(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels=1, filters=64, num_res_blocks=16, num_upsample=2) -> None:
        super(GeneratorRRDB, self).__init__()
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_upsample = num_upsample

        self.conv1 = nn.Conv2d(self.channels, filters, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDensBlock(filters) for _ in range(self.num_res_blocks)]
        )

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        upsample_layers = []
        for _ in range(self.num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            ]

        self.upsampling = nn.Sequential(*upsample_layers)

        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, self.channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, self.channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        return out 

if __name__ == "__main__":
    # print(GeneratorRRDB(channels=16))
    pass 