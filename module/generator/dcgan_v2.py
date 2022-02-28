import argparse

import torch
import torch.nn as nn
import torch.optim as optim


## some parsers for control hyperparam
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150, help="number of epochs of training")
parser.add_argument('--dropout_rate', type=float, default=0.5, help="set droput rate")
parser.add_argument('--leakyrelu_rate', type=float, default=0.2, help="set LeakyReLU learning rate")
parser.add_argument('--z', type=int, default=100, help='generator\'s input noise')
option = parser.parse_args()

print(option)


class DCGenerator(nn.Module):

    def __init__(self, generator_channels=3, conv_trans_strides=1, 
                generator_filters=64, kernel_size=4, conv_trans_paddings=0, z=100) -> None:
        super(DCGenerator, self).__init__()
        self.generator_channels = generator_channels
        self.conv_trans_strides = conv_trans_strides
        self.generator_filters = generator_filters
        self.kernel_size = kernel_size
        self.conv_trans_paddings = conv_trans_paddings
        self.z = z 



        def dc_generator_block(
            self, in_filters: int, 
            out_filters: int, 
            first_block = False) -> list:
            layers = []

            layers.append(nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=self.kernel_size,
                stride=self.conv_trans_strides,
                padding=self.conv_trans_paddings
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
                first_block = (i == 0)
            ))

            input = output

        layers.append(nn.ConvTranspose2d(
            in_channels = input,
            out_channels = self.generator_channels,
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



class DCDiscriminator(nn.Module):

    def __init__(
        self, discriminator_channels=3, 
        discriminator_kernels=4, discriminator_strides=2, 
        discriminator_paddings=1) -> None:

        super(DCDiscriminator, self).__init__()
        self.discriminator_channels = discriminator_channels
        self.discriminator_kernels = discriminator_kernels
        self.discriminator_strides = discriminator_strides
        self.discriminator_paddings = discriminator_paddings
        self.leaky_relu_rate = option.leakyrelu_rate
        self.dropout_rate = option.dropout_rate

        def dc_discriminator_block(self, in_filters: int, out_filters: int) -> list:
            layers = []
            layers.append(nn.Conv2d(
                in_channels = in_filters, 
                out_channels = out_filters, 
                kernel_size = self.discriminator_kernels, 
                stride = self.discriminator_strides, 
                padding = self.discriminator_paddings, 
                bias = False))

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(self.leaky_relu_rate, inplace=True))
            layers.append(nn.Dropout(self.dropout_rate))

            return layers

        # generate discriminator's layers
        CONV_INPUT: int = 64

        layers = []
        layers.append(nn.Conv2d(
            in_channels = self.discriminator_channels, 
            out_channels = CONV_INPUT,
            kernel_size = self.discriminator_kernels,
            stride = self.discriminator_strides,
            padding = self.discriminator_paddings))
        layers.append(nn.Sigmoid())
        
        input = CONV_INPUT
        for _, output in enumerate([64, 128, 256, 512]):
            layers.extend(dc_discriminator_block(
                self, 
                in_filters = input,
                out_filters = output))
            
            output = input 
        
        self.models = nn.Sequential(*layers)
        print(self.models)

### RUN TEST ### 

def main() -> any:
    EPOCHS: int = option.epochs
    IMG_SIZE: int = 64 
    BATCH_SIZE: int = 64

    DCGenerator()
    DCDiscriminator()



if __name__ == "__main__":
    main()
