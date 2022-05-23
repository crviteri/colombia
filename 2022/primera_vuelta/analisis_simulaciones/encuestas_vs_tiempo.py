#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:38:33 2022

@author: crviteri
"""


import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
lista = plt.style.available
plt.style.use([lista[11],lista[0]])
#plt.style.use('default')


config_file = json.load(open('presidente/config_presidente.json', 'r'))
encuestadoras_atributos = json.load(open('encuestadoras.json', 'r'))

fecha_eleccion = datetime.strptime(config_file['fecha eleccion'], '%m/%d/%y')
csv_fname = config_file['base con encuestas'] + '.csv'
papeleta = config_file['papeleta']
resultados = config_file['resultados eleccion']

candidatos_ordenados = sorted(resultados, key=lambda k: resultados[k]['fraccion validos'], reverse=True)

suma = 0
for candidato in resultados:
    if candidato not in candidatos_ordenados[0:5]:
        suma += resultados[candidato]['fraccion validos']

resultados['Otros'] = {'1er apellido': 'Otros', 'fraccion validos': suma}

candidatos = candidatos_ordenados[0:5] + ['Otros']

dfh_todas = pd.read_csv(csv_fname)
dfh_todas['Fecha'] = pd.to_datetime(dfh_todas['Fecha de campo'], format='%d/%m/%Y')
dfh_todas = dfh_todas.sort_values(by='Fecha', ascending=True)
fecha_i = dfh_todas['Fecha'].min().strftime('%d/%m/%Y')
dfh_todas['Fecha'] = dfh_todas['Fecha'].dt.strftime('%d/%m/%Y')

encuestadoras = dfh_todas['Encuestadora'].unique().tolist()
# encuestadoras.remove(nan)
encuestadoras.sort()


### Plots grouped by candidates
#color_map = plt.cm.gist_ncar(linspace(0, 1, len(encuestadoras)))
fig1, axarr = plt.subplots(int(ceil(len(candidatos)/2.0)), 2, num='encuestadoras', sharex=True, figsize=(6, 6))

j = 0
x = {}
x_no_na = {}
for candidato in candidatos:
    k = 0
    for encuesta in encuestadoras:
        if j == 0:
            x[encuesta] = {}
            x_no_na[encuesta] = {}
        dfh = dfh_todas[dfh_todas['Encuestadora'] == encuesta]
        if 'Si' in dfh['Cumple criterios'].values:
            dfh = dfh[dfh['Cumple criterios'] == 'Si']
            fecha = array([datetime.strptime(i,'%d/%m/%Y') for i in dfh['Fecha']])
            dfh['Margen de error'] = [float(i.replace(',', '.').strip('%')) for i in dfh['Margen de error']]
            m = int(floor(j/2))
            n = mod(j, 2)
            if candidato is not 'Otros':
                dfh[candidato] = dfh[candidato].fillna('nan%')
                dfh[candidato] = [i.replace(',', '.') for i in dfh[candidato]]
                dfh['Total válidos'] = [i.replace(',', '.') for i in dfh['Total válidos']]
                x[encuesta][candidato] = dfh[candidato].str[:-1].astype('float64') \
                                         /dfh['Total válidos'].str[:-1].astype('float64')
                x_no_na[encuesta][candidato] = x[encuesta][candidato].fillna(0)
            else:
                x_df = pd.DataFrame.from_dict(x_no_na[encuesta])
                x[encuesta][candidato] = 1.0-x_df.sum(axis=1)
            axarr[m, n].errorbar(fecha, 100.0*x[encuesta][candidato], dfh['Margen de error'], fmt='o-', markersize=5,
                                 color=encuestadoras_atributos[encuesta]['color'], label=encuesta)
            axarr[m, n].grid(True)
            if j < len(candidatos) - 2:
                axarr[m,n].set_xticklabels([])
            axarr[m, n].set_ylim(0, 50)
            axarr[m, n].set_yticks([0, 10, 20, 30, 40, 50])
            axarr[m, n].text(0.025,0.9,resultados[candidato]['1er apellido'], horizontalalignment='left', transform=axarr[m,n].transAxes, fontsize=14)
            #        axarr[m,n].set_title(candidato)
            axarr[m, n].set_ylabel('%')
            k += 1
    # axarr[m, n].scatter(fecha_eleccion, 100.0*resultados[candidato]['fraccion validos'], marker="<", label='resultado JNE', color='k', linewidths=3)
    axarr[m, n].vlines(fecha_eleccion, 0, 100, linestyles='--', label=u'fecha elección')
    j += 1
fig1.autofmt_xdate()
axarr[m, n].set_xlim(datetime.strptime(fecha_i, '%d/%m/%Y'), fecha_eleccion)
x_lims = axarr[m, n].get_xlim()
axarr[m, n].set_xlim(x_lims[0]-5,x_lims[1]+5)
# axarr[m,n+1].set_axis_off()
# axarr[m, n].set_ylim(0, 100)
# axarr[m, n].set_yticks([0, 25, 50, 75, 100])
axarr[m, n].legend(bbox_to_anchor=(0.8, -0.5), ncol=4, numpoints=1, fontsize='x-small')
# fig1.suptitle(u'Elecciones Presidenciales Perú 2021. \nEstimaciones del voto válido publicadas por encuestadoras. \n(Realizado por Cálculo Electoral.)', fontsize=9, x=.05, ha='left', weight='bold')
# fig1.suptitle(u'Evolución de las estimaciones del voto válido publicadas por encuestadoras.'
#               u'\nRealizado por Cálculo Electoral.', fontsize=9, x=.05, ha='left', weight='bold')
# fig1.suptitle(u'Resultados JNE y evolución del voto válido publicado por encuestadoras.'
#               u'\nRealizado por Cálculo Electoral.', fontsize=9, x=.05, ha='left', weight='bold')
plt.show()
plt.savefig(config_file['grafico_encuestas_png'] + '.png', dpi=330)
