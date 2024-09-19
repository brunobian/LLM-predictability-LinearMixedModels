
# 20 de agosto
## running model.py (maj version)

```
11000 of 11954: 92.019%Data bunch crated
epoch     train_loss  valid_loss  accuracy  time    
0         3.906703    3.812281    0.349362  1:43:52   
1         3.505440    3.471526    0.371139  1:44:06   
2         3.410365    3.376447    0.377568  1:44:09   
3         3.390575    3.311733    0.383338  1:44:03   
4         3.304545    3.264617    0.387852  1:44:15   
5         3.260852    3.223455    0.392012  1:44:05   
6         3.276509    3.187599    0.396304  1:44:04   
7         3.227356    3.154278    0.400059  1:44:04   
8         3.191780    3.135717    0.402396  1:44:02   
9         3.177944    3.130546    0.403084  1:44:04 
```

## cantidad de tokens (maj version)

- Cantidad de tokens wikipedia: 2751415
- Cantidad de tokens cuentos:    535068
- Cantidad de tokens comunes:   2969858

## running fine_tuning.py (maj version)
```
1         4.754032    4.344392    0.232268  1:10:43   
epoch     train_loss  valid_loss  accuracy  time    
0         4.243888    4.114587    0.254450  1:19:31   
1         4.215551    4.088652    0.257867  1:19:31   
2         4.149891    4.041308    0.260940  1:19:28   
3         4.104953    4.003483    0.264977  1:19:37   
4         4.055195    3.964124    0.268966  1:19:28   
5         4.039286    3.913886    0.272971  1:19:30   
6         4.024626    3.881094    0.276518  1:19:28   
7         4.032934    3.869521    0.277549  1:19:25
```

>> y si probamos `dict(learn.model.named_modules())['1.decoder'].reset_parameters`?