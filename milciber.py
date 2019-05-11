#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:35:20 2019

@author: alopes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import weibull_min
#%%
pais=np.array([('EUA',0.0615,0.6364,0.8233,0.922),('Russia',0.0639,0.6494,0.6749,0.816),('China',0.0673,0.3506,0.5800,0.752),('India',0.1065,0.5065,0.4229,0.640),\
               ('Franca',0.1584,0.8312,0.7906,0.901),('Japao',0.1707,0.6234,0.8215,0.909),('Coreia do Sul',0.1761,0.3896,0.4519,0.903),('Reino Unido',0.1797,0.7532,0.8396,0.922),\
               ('Turquia',0.2089,0.4156,0.6183,0.791),('Alemanha',0.2097,0.8052,0.8195,0.936),('Italia',0.2277,0.7662,0.6663,0.880),('Egito',0.2283,0.3506,0.4958,0.696),\
               ('Brasil',0.2487,0.4675,0.5917,0.759),('Indonesia',0.2804,0.1948,0.5022,0.694),('Israel',0.2964,0.6494,0.7797,0.903),\
               ('Paquistao',0.3087,0.2987,0.3639,0.562),('Australia',0.3277,0.5974,0.8049,0.939),\
               ('Espanha',0.3921,0.8961,0.7324,0.891),('Canada',0.3941,0.5714,0.7885,0.926),('Vietna',0.3988,0.3377,0.5001,0.694),\
               ('Polonia',0.4059,0.7013,0.6659,0.865),('Arabia Saudita',0.4286,0.5844,0.6764,0.853),('Tailandia',0.4302,0.4286,0.5835,0.755),('Argelia',0.4551,0.1429,0.4621,0.754),\
               ('Grecia',0.4955,0.7662,0.6544,0.870),('Ucrania',0.5082,0.6364,0.5810,0.751),('Rep Tcheca',0.5119,0.9091,0.6937,0.888),('Suecia',0.5263,0.5714,0.8348,0.933),\
               ('Africa do Sul',0.5405,0.2727,0.5480,0.699),('Suica',0.5459,0.7273,0.8513,0.944),('Mexico',0.5574,0.3636,0.5437,0.774),('Holanda',0.6063,0.6623,0.8388,0.931),\
               ('Noruega',0.6103,0.6234,0.8378,0.953),('Argentina',0.6274,0.4286,0.6109,0.825),('Bielorussia',0.6418,0.5325,0.7550,0.808),\
               ('Romenia',0.6461,0.7143,0.6169,0.811),('Malasia',0.6523,0.7273,0.6690,0.802),('Peru',0.6841,0.4286,0.5139,0.750),('Venezuela',0.6931,0.3247,0.5014,0.761),\
               ('Bangladesh',0.7156,0.2857,0.3622,0.608),('Colombia',0.7157,0.4675,0.5609,0.747),('Etiopia',0.7361,0.3247,0.3039,0.463),\
               ('Usbequistao',0.7365,0,0,0.710),('Bulgaria',0.7525,0.5195,0.6359,0.813),('Dinamarca',0.7767,0.8182,0.8355,0.929),\
               ('Azerbaijao',0.7791,0.3377,0.6171,0.757),('Cazaquistao',0.79080,0.4545,0.6681,0.800),\
               ('Austria',0.8007,0.5974,0.8049,0.908),('Chile',0.8054,0.5584,0.6571,0.843),('Singapura',0.8161,0.8052,0.8311,0.932),('Eslovaquia',0.8181,0.7922,0.6673,0.855),\
               ('Marrocos',0.8244,0.2338,0.5171,0.667),('Finlandia',0.8661,0.7922,0.8226,0.920),('Filipinas',0.8862,0.3247,0.5192,0.699),\
               ('Bolivia',0.9139,0.2857,0.4512,0.693),('Equador',0.9451,0.3247,0.5206,0.752),('Portugal',0.9611,0.6883,0.7065,0.847),('Belgica',0.9633,0.6494,0.7762,0.916),\
               ('Sudao',1.0051,0.0649,0.2550,0.502),('Croacia',1.0067,0.5714,0.6691,0.831),('Cuba',1.0565,0.1299,0.2910,0.777),\
               ('Afeganistao',1.1511,0.1558,0.1950,0.498),\
               ('Zambia',1.3035,0.4156,0.3536,0.588),('Servia',1.3041,0.6364,0.6162,0.787),('Tunisia',1.3322,0.3117,0.5196,0.735),('Lituania',1.3751,0.8831,0.7095,0.858),\
               ('Oman',1.3842,0.3247,0.6286,0.821),('Zimbabue',1.3906,0.1558,0.3603,0.535),('Georgia',1.4098,0.6494,0.5966,0.780),('Quenia',1.4177,0.3377,0.4169,0.590),\
               ('Nova Zelandia',1.5184,0.5584,0.8094,0.917),('Eslovenia',1.5186,0.5714,0.7047,0.896),('Mongolia',1.5447,0.1169,0.5551,0.741),('Sri Lanka',1.5687,0.2468,0.4955,0.770),\
               ('Albania',1.5783,0.4156,0.5356,0.785),('Uganda',1.5802,0.5065,0.3309,0.516),('Tadjiquistao',1.6031,0.1429,0.4714,0.650),\
               ('Chade',1.6041,0.1818,0.2206,0.404),('Armenia',1.6888,0.2597,0.5951,0.755),('Mali',1.7172,0.1948,0.3151,0.427),\
               ('Botsuana',1.7786,0.1429,0.4795,0.717),('Guatemala',1.8203,0.2857,0.4175,0.650),('Camaroes',1.8242,0.2208,0.3333,0.556),('Moldavia',1.8275,0.4805,0.6082,0.700),\
               ('Costa do Marfim',1.8514,0.2338,0.3939,0.492),('Letonia',1.8691,0.7143,0.6169,0.847),('Catar',1.8696,0.5584,0.7319,0.856),\
               ('Gana',1.9231,0.2078,0.4525,0.592),('Uruguai',1.9867,0.4805,0.6794,0.804),('Honduras',2.0554,0.1039,0.4283,0.617),\
               ('Estonia',2.1115,0.9091,0.7927,0.871),('Paraguai',2.1503,0.3896,0.4519,0.702),('Irlanda',2.1579,0.6364,0.7796,0.938),\
               ('Nigeria',2.1631,0.4416,0.3586,0.532),('Nicaragua',2.2228,0.2208,0.3635,0.658),('Montenegro',2.3173,0.3377,0.6291,0.814),\
               ('Nepal',2.3472,0.1429,0.3726,0.574),('Rep do Congo',2.4035,0.0260,0.1550,0.457),('Bosnia e Herz',2.5483,0.4935,0.5266,0.768),\
               ('Madagascar',2.6168,0.0260,0.2697,0.519),('Rep Dominicana',2.6208,0.4156,0.4826,0.736),('Laos',2.7113,0.1688,0.3884,0.601),\
               ('El Salvador',2.8716,0.1948,0.4553,0.674),\
               ('Panama',3.5849,0.4805,0.5526,0.789),('Suriname',3.8295,0.0649,0.5150,0.720),\
               ('Liberia',4.8442,0.1948,0.4000,0.435),('Butao',6.3988,0.1299,0.4559,0.612)])
#%%
nome=[]
milpow=[]
cibsec=[]
digdev=[]
idh=[]
for k in range (len(pais)):
    nome.insert(k,pais[k][0])
    milpow.insert(k,pais[k][1])
    cibsec.insert(k,pais[k][2])
    digdev.insert(k,pais[k][3])
    idh.insert(k,pais[k][4])
milpow=np.float_(milpow)
milpow=np.divide(1,milpow)
cibsec=np.float_(cibsec)
digdev=np.float_(digdev)
idh=np.float_(idh)
dados=pd.DataFrame({'Poder Militar':milpow,'Ciber Seg':cibsec,'Dsv Digital':digdev,'IDH':idh})
corr=dados.corr()
#%%
plt.matshow(corr)
plt.xticks(range(len(dados.columns)), dados.columns)
plt.yticks(range(len(dados.columns)), dados.columns)
plt.colorbar()
plt.show()
#%%
plt.figure()
plt.title('Cor/tamanho: mais claro => maior IDH')
plt.xlabel('Poder Militar')
plt.ylabel('Segurança Cibernética')
plt.scatter(milpow,cibsec,c=idh,s=idh*100)
plt.show()
#%%
plt.figure()
plt.title('Cor/tamanho: mais claro => maior Dsv Digital')
plt.xlabel('Poder Militar')
plt.ylabel('IDH')
plt.scatter(milpow,idh,c=digdev,s=digdev*100)
plt.show()
#%%
plt.figure()
plt.title('Cor/tamanho: mais claro => maior Seg Ciber')
plt.xlabel('Dsv Digital')
plt.ylabel('IDH')
plt.scatter(digdev,idh,c=cibsec,s=cibsec*100)
plt.show()
#%%
plt.figure()
loc,scale=expon.fit(np.divide(1,milpow))
mu_mil=expon.mean(loc=loc,scale=scale)
plt.hist(np.divide(1,milpow),density=1,label='Media=%.2f'%mu_mil)
x=np.linspace(min(np.divide(1,milpow)),max(np.divide(1,milpow)))
plt.plot(x,expon.pdf(x,loc=loc,scale=scale))
plt.title('Poder Militar')
plt.legend(loc='best')
plt.show()
#%%
plt.figure()
c,loc,scale=weibull_min.fit(cibsec)
mu_cib=weibull_min.mean(c,loc=loc,scale=scale)
plt.hist(cibsec,density=1,label='Media=%.2f'%mu_cib)
x_cib=np.linspace(min(cibsec),max(cibsec))
plt.plot(x_cib,weibull_min.pdf(x_cib,c,loc=loc,scale=scale))
plt.title('Segurança Cibernética')
plt.legend(loc='best')
plt.show()
#%%
plt.figure()
c_dsv,loc_dsv,scale_dsv=weibull_min.fit(digdev)
mu_dsv=weibull_min.mean(c_dsv,loc=loc_dsv,scale=scale_dsv)
plt.hist(digdev,density=1,label='Media=%.2f'%mu_dsv)
x_dsv=np.linspace(min(digdev),max(digdev))
plt.title('Desenvolvimento Digital')
plt.plot(x_dsv,weibull_min.pdf(x_dsv,c_dsv,loc=loc_dsv,scale=scale_dsv))
plt.legend(loc='best')
plt.show()