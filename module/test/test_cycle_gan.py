from cycle import CycleDiscriminator, CycleGenerator
import torch 
import unittest


class TestCycleGANModule(unittest.TestCase):
    def test_discriminator(self):
        x = torch.randn((5, 3, 256, 256))
        model = CycleDiscriminator(in_channels=3)
        pred = model(x)
        self.assertEqual(pred.shape, torch.Size([5, 1, 30, 30]))
    
    def test_generator(self):
        img_channels = 3
        img_size = 256 
        x = torch.randn((2, img_channels, img_size, img_size))
        gen = CycleGenerator(img_channels, num_residual=9)
        self.assertEqual(gen(x).shape, torch.Size([2, 3, 256, 256]))
