# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



################# UNet #########################
#											   #	
# Adapted from TP Segmentation, DL4MI MVA 2023 #
#     										   #	
################################################

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):  return getattr(nn, activation_type)()
    else:  return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """This block implements the sequence: (convolution => [BN] => ReLU)"""  
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
      
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv-1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class DownBlock(nn.Module):
    """Downscaling with maxpooling and convolutions"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)  

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(Bottleneck, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = nn.Sequential(ConvBatchNorm(in_channels, 1024, activation),
                                    ConvBatchNorm(1024, out_channels, activation))

    def forward(self, input):
        out = self.maxpool(input)
        out = self.nConvs(out)
        return out

class UpBlock(nn.Module):
    """Upscaling then conv"""
    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock, self).__init__()        
        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1) # dim 1 is the channel dimension
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        '''
        n_channels : number of channels of the input. 
                        
        n_labels : number of channels of the ouput.
                      By default 4 (3 labels + 1 for the background)
        '''
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = _make_nConv(self.in_channels, 64, nb_Conv=2, activation='ReLU')
        self.down2 = DownBlock(64, 128, nb_Conv=2)
        self.down3 = DownBlock(128, 256, nb_Conv=2)
        self.down4 = DownBlock(256, 512, nb_Conv=2)

        self.Encoder = [self.down1, self.down2, self.down3, self.down4]

        self.bottleneck = Bottleneck(512, 512)

        self.up1 = UpBlock(1024, 256, nb_Conv=2)
        self.up2 = UpBlock(512, 128, nb_Conv=2)
        self.up3 = UpBlock(256, 64, nb_Conv=2)
        self.up4 = UpBlock(128, 64, nb_Conv=2)

        self.Decoder = [self.up1, self.up2, self.up3, self.up4]

        self.outc = nn.Conv2d(64, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        # self.last_activation = get_activation('Softmax')

        self.history = {'train_loss' : [],
                        'val_loss' : []}

    
    def forward(self, x):
        # Forward 
        skip_inputs = []

        # Forward through encoder
        for i, block in enumerate(self.Encoder):
            x = block(x)  
            skip_inputs += [x] 

        # We are at the bottleneck.
        bottleneck = self.bottleneck(x)

        # Forward through decoder
        skip_inputs.reverse()

        decoded = bottleneck
        for i, block in enumerate(self.Decoder):
            # Concat with skipconnections
            skipped = skip_inputs[i]
            decoded = block(decoded, skipped)
        out = self.outc(decoded)
        return out




################# DCNN ########################
# Adapted from
# https://github.com/LSL000UD/RTDosePrediction/
# blob/main/RTDosePrediction/Src/DCNN/model.py
##############################################

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DenseFeaureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DenseFeaureAggregation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=1 * in_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, base_ch, dilation=2, kernel_size=3, padding=2, stride=1, bias=True),

        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + base_ch, base_ch, dilation=3, kernel_size=3, padding=3, stride=1, bias=True),

        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 2 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 2 * base_ch, base_ch, dilation=5, kernel_size=3, padding=5, stride=1, bias=True),

        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 3 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 3 * base_ch, base_ch, dilation=7, kernel_size=3, padding=7, stride=1, bias=True),

        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 4 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 4 * base_ch, base_ch, dilation=9, kernel_size=3, padding=9, stride=1, bias=True),

        )

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 5 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 5 * base_ch, out_ch, dilation=1, kernel_size=1, padding=0, stride=1, bias=True),
        )

    def forward(self, x):
        out_ = self.conv1(x)
        concat_ = torch.cat((out_, x), dim=1)
        out_ = self.conv2(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv3(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv4(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv5(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv_out(concat_)
        return out_


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )

        self.DFA = DenseFeaureAggregation(list_ch[4], list_ch[4], list_ch[4])

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_4 = self.DFA(out_encoder_4)
        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]


class Decoder(nn.Module):
    def __init__(self, out_ch, list_ch):
        super(Decoder, self).__init__()

        self.upconv_3_1 = nn.ConvTranspose2d(list_ch[4], list_ch[3], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2_1 = nn.ConvTranspose2d(list_ch[3], list_ch[2], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1_1 = nn.ConvTranspose2d(list_ch[2], list_ch[1], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder

        out_decoder_3_1 = self.decoder_conv_3_1(
            torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2_1 = self.decoder_conv_2_1(
            torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1)
        )
        out_decoder_1_1 = self.decoder_conv_1_1(
            torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1)
        )

        output = self.conv_out(out_decoder_1_1)
        return [output]


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch):
        super(Model, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(out_ch, list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder[0]




################# R2U-Net / Attention U-Net #########################
#											   						#	
# Adapted from 
# https://github.com/LeeJunHyun/Image_Segmentation 						#
#     										   						#	
#####################################################################



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
        #self.history = {'train_loss' : [],
        #                'val_loss' : []}


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
