import torch
import torch.nn as nn
import torch.optim as optim 


class DCGenerator(nn.Module):
    def __init__(self, generator_channels=3, conv_trans_strides=1, generator_filters=64, 
                        kernel_size=4, conv_trans_paddings=0, z=100) -> None:
        super(DCGenerator, self).__init__()
        self.generator_channels = generator_channels
        self.conv_trans_strides = conv_trans_strides
        self.generator_filters = generator_filters
        self.kernel_size = kernel_size
        self.conv_trans_paddings = conv_trans_paddings

        self.z = z                  # noise 

        self.img_size = 64
        self.batch_size = 64
    
        def dc_generator_block(self, in_filters: int, out_filters: int, first_block = False) -> list:
            layers = []

            layers.append(nn.ConvTranspose2d(
                in_channels = in_filters, 
                out_channels = out_filters, 
                kernel_size = self.kernel_size, 
                stride = self.conv_trans_strides,
                padding = self.conv_trans_paddings
            ))

            layers.append(nn.BatchNorm2d(self.generator_filters))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout())

            return layers
        
        ### generate layers ### 
        layers = []
        input = self.z

        for i, output in enumerate([512, 256, 128, 64]):
            layers.extend(dc_generator_block(
                    self, 
                    in_filters = input, 
                    out_filters = output, 
                    first_block = (i==0)
                ))

            input = output

        layers.append(nn.ConvTranspose2d(
                in_channels = input, 
                out_channels = self.channels, 
                kernel_size = self.kernel_size, 
                stride = 2, 
                padding = 1
            ))

        layers.append(nn.Tanh()) 
        
        self.models = nn.Sequential(*layers)

        print(self.models)

    
    def forward(self, input) -> any:
        output = self.models(input)
        return output

    ## build model -> return 

class DCDiscriminator(nn.Module):
    def __init__(self, discriminator_channels=3, leaky_relu_rate=0.2) -> None:
        super(DCDiscriminator, self).__init__()
        self.discriminator_channels = discriminator_channels
        self.leaky_relu_rate = leaky_relu_rate 
    
        def dc_discriminator_block(self) -> list:
            layers = []
            layers.append(nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout())

            return layers 
        
        ## generate discriminator's layers 
        layers = []
        
        for i, output in enumerate([64, 128, 256, 512]):
            ...
        
    




if __name__ == "__main__":
    DCGenerator()
