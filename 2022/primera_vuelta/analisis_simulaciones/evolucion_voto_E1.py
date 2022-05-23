#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:02:22 2022

@author: crviteri
"""


import csv
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import allantools


def find_top_three(encuesta, candidatos):
    validos = {candidato: encuesta[candidato] for candidato in candidatos}
    solo_validos = {}
    for e in validos:
        if validos[e] != '':
            solo_validos[e] = validos[e]
    return sorted(solo_validos, key=solo_validos.get, reverse=True)[:3]


def count_empty_entries(encuesta, candidatos):
    i = 0
    for candidato in candidatos:
        if encuesta[candidato] == '':
            i += 1
    return i


def parameterize_decision(dias, decision, exp_plus_offset, config_file):
    x_d = [i for i in dias if i is not nan]
    y_d = [i for i in decision if i is not nan]
    x_d.append(0)
    y_d.append(1.0)
    p, pcov = curve_fit(exp_plus_offset, x_d, y_d, p0 = array([30.0, 0.4]), bounds=([1.0, 0], [np.inf, 1.0]))
    scatter(x_d,y_d)
    x_ = linspace(x_d[0], x_d[-1], abs(int(x_d[0])))
    plot(x_, exp_plus_offset(x_, p[0], p[1]), 'r')
    plt.xlabel(u'días a la elección')
    plt.ylabel(u'nivel de decisión')
    plt.title('tau = {:.1f} days, offset = {:.2f}'.format(p[0],p[1]))
    plt.show()
    plt.savefig(config_file['parametrizacion decision'] + '.eps', dpi=300)
    return p[0], p[1]


def adev_evolucion_voto(x, y):
    fx = 1/mean(diff(x))
    b = allantools.Plot()

    allan = allantools.Dataset(data=y, rate=fx, data_type='freq', taus=linspace(1,len(y),len(y))/fx)
    allan.compute('oadev')
    sigma0_ref = allan.out["stat"][0] / sqrt(linspace(1, len(y), len(y)))

    # b.ax.errorbar(allan.out["taus"], allan.out["stat"], yerr=allan.out["stat_err"], label=allan.out["stat_id"])
    # b.ax.plot(linspace(allan.out["taus"][0], len(x) * mean(diff(x)), len(x)), sigma0_ref, '--',
    #           label='$\sigma$(0)/$\sqrt{N}$')
    # b.ax.set_xlabel(r'$\tau$ en días')
    b.ax.errorbar(linspace(1, len(allan.out["taus"]), len(allan.out["taus"])), allan.out["stat"],
                  yerr=allan.out["stat_err"], label=allan.out["stat_id"])
    b.ax.plot(linspace(1, len(y), len(y)), sigma0_ref, '--', label='$\sigma$(0)/$\sqrt{N}$')
    b.ax.set_xlabel(r'$N$')
    # b.ax.set_ylabel(allan.out["stat_id"])
    b.ax.set_ylabel(r'$\sigma$($\tau$)')
    b.ax.grid(grid, which="minor", ls="-", color='0.65')
    b.ax.grid(grid, which="major", ls="-", color='0.25')
    b.ax.legend(loc=1)
    b.show()


config_file = json.load(open('presidente/config_presidente.json', 'r'))

lista = plt.style.available
plt.style.use([lista[11],lista[0]])

lowess = sm.nonparametric.lowess

exp_plus_offset = lambda x, tau, offset: (1 - offset) * exp((1.0 / tau)*x) + offset

csv_fname = config_file['base con encuestas'] + '.csv'  # archivo con compilacion de encuestas
encuestadoras = json.load(open('encuestadoras.json', 'rb'))  # archivo con calificaciones de encuestadoras
fecha_eleccion = config_file['fecha eleccion']

papeleta = config_file['papeleta']

candidatos = list(papeleta.keys())

eleccion_ut = time.mktime(time.strptime(fecha_eleccion, '%m/%d/%y'))  # Unix Timestamp
for encuestadora in encuestadoras:
    encuestadoras[encuestadora]['frecuencia'] = 0
encuestas = []
with open(csv_fname, 'r') as csvfile:
    # for i in range(1): # skip first row
    #     csvfile.next()
    file_read = csv.DictReader(csvfile)
    for row in file_read:
        #if np.float(row['SUMA']) == 1.0:
        if row['Fecha de campo'] != '' and row['Cumple criterios'] == 'Si':
            encuestas.append(row)
            encuestas[-1]['Dias'] = -(eleccion_ut - time.mktime(time.strptime(encuestas[-1]['Fecha de campo'],
                                                                              '%d/%m/%Y'))) / (24 * 3600)
            encuestadoras[encuestas[-1]['Encuestadora']]['frecuencia'] += 1
encuestas.sort(key=lambda k: k['Dias'], reverse=False)

# Convert a string percent to a float
for encuesta in encuestas:
    for candidato in candidatos:
        if encuesta[candidato] != '':
            encuesta[candidato] = np.float(encuesta[candidato].strip('%').
                                           replace(',', '.'))/100.0


# If all possible candidates were NOT reported by pollster, then assign missing fraction to all other candidates not reported.
for encuesta in encuestas:
    empties = count_empty_entries(encuesta, candidatos)
    for candidato in candidatos:
        if float(encuesta['Suma'].strip('%').replace(',', '.')) < 100.0 and \
                float(encuesta['Total válidos'].strip('%').replace(',', '.')) == \
                float(encuesta['Total Papeleta'].strip('%').replace(',', '.')):
            if encuesta[candidato] == '':
                encuesta[candidato] = (1 - float(encuesta['Total'].strip('%').replace(',', '.'))/100.0) / empties


# Suma unicamente candidatos en la lista. Se asume que solo una fraccion de 'Blanco' votara de esa manera.
div_blanco = 2
sumas = []
for encuesta in encuestas:
    suma = 0
    for candidato in candidatos:
        try:
            if candidato == 'Blanco':
                suma = suma + encuesta[candidato]/div_blanco
            else:
                suma = suma + encuesta[candidato]
        except:
            suma = suma
    sumas.append(suma)


# Distributes others (1-suma) equally among top candidates.
homologadas = []
asignaciones = []
participacion = []
decision = []
dias = []
N_candidatos = len(candidatos)
candidatos_sin_blanco = [i for i in candidatos if i != 'Blanco']
ind = 0
for encuesta in encuestas:
    homologada = []
    asignacion_otros = []
    top_three = find_top_three(encuesta, candidatos_sin_blanco)
    N_grupos_asignacion = len(top_three) + 1  # there are not always top_n elements in the list
    for candidato in candidatos:
        # assume half of 'Blanco' and candidates NOT on ballot (but in poll) will follow bandwagon effect
        if candidato in top_three:
            asignacion_otros.append((1 - sumas[ind]) / N_grupos_asignacion)
        else:
            asignacion_otros.append((1 - sumas[ind]) / (N_grupos_asignacion*(N_candidatos - N_grupos_asignacion)))
        if encuesta[candidato] != '':
            if candidato == 'Blanco':
                homologada.append(np.float(encuesta[candidato]) / div_blanco)
            else:
                homologada.append(np.float(encuesta[candidato]) + asignacion_otros[-1])
        else:
            if candidato == 'Blanco':
                homologada.append(nan)
            else:
                homologada.append(asignacion_otros[-1])
    homologadas.append(dict(zip(candidatos, homologada)))
    asignaciones.append(dict(zip(candidatos, asignacion_otros)))
    if encuesta['Participación'] != '':
        participacion.append(np.float(encuesta['Participación'].strip('%').replace(',', '.'))/100.0)
    else:
        participacion.append(mean(config_file['participacion historica 1era']))
    if encuesta['Indecisos'] != '':
        decision.append(1.0-np.float(encuesta['Indecisos'].strip('%').replace(',', '.'))/100.0)
        dias.append(encuesta['Dias'])
    elif encuesta['No sabe/No responde'] != '':
        decision.append(1.0 - np.float(encuesta['No sabe/No responde'].strip('%').replace(',', '.')) / 100.0)
        dias.append(encuesta['Dias'])
    else:
        decision.append(nan)
        dias.append(nan)
    ind += 1


(tau, offset) = parameterize_decision(dias, decision, exp_plus_offset, config_file)  # filter time constant in days
# (tau, offset) = (15, 0.45)

# Figure with Decision
x_d = linspace(min(dias), max(dias), int(max(dias)-min(dias)))
fig_d = plt.figure()
ax_d = plt.subplot(111)
ax_d.scatter(dias, 100*array(decision), color='#ffffff')
ax_d.plot(array(x_d), 100.0*exp_plus_offset(array(x_d), tau, offset), 'k', linewidth=5)
y_lims = ax_d.get_ylim()
ax_d.set_ylim((y_lims[0]), 100)
ax_d.set_xlim(1.02*min(x_d), 0)
ax_d.set_xlabel(u'días a la elección')
ax_d.set_ylabel(u'% nivel de decisión')
ax_d.grid()
plt.show()
fig_d.savefig(config_file['evolucion decision']+'.png', dpi=330)


# Lista de diccionarios con cada encuesta en votos validos
bd_homologadas = []
ind = 0
for encuesta in homologadas:
    voto_valido = []
    for candidato in candidatos:
        voto_valido.append(encuesta[candidato])
    bd_homologadas.append(dict(zip(candidatos, voto_valido)))
    bd_homologadas[ind]['Participacion'] = participacion[ind]
    bd_homologadas[ind]['Asignaciones'] = asignaciones[ind]
    bd_homologadas[ind]['Decision'] = decision[ind]
    bd_homologadas[ind]['Fecha'] = encuestas[ind]['Fecha de campo']
    bd_homologadas[ind]['Dias'] = encuestas[ind]['Dias']
    bd_homologadas[ind]['Encuestadora'] = encuestas[ind]['Encuestadora']
    if encuestas[ind]['Representatividad'] != '':
        bd_homologadas[ind]['Representatividad'] = float(encuestas[ind]['Representatividad'].strip('%').replace(',', '.'))/100.0
    else:
        bd_homologadas[ind]['Representatividad'] = 0.0
    if encuestas[ind]['Ponderación metodología'] != '':
        bd_homologadas[ind]['metodologia'] = float(encuestas[ind]['Ponderación metodología'].strip('%').replace(',', '.'))/100.0
    else:
        bd_homologadas[ind]['metodologia'] = 0.0
    bd_homologadas[ind]['Peso'] = exp((1.0 / tau) * encuestas[ind]['Dias']) * (
            bd_homologadas[ind]['Representatividad'] + encuestadoras[encuestas[ind]['Encuestadora']][
        'acierto'] +
            bd_homologadas[ind]['metodologia']) / encuestadoras[encuestas[ind]['Encuestadora']]['frecuencia']
    ind += 1


# Archivo con encuestas homologadas
keys = bd_homologadas[0].keys()
with open(config_file['encuestas homologadas'] + '.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(bd_homologadas)


# Diccionario de candidatos con evolucion de promedio ponderado de intencion de voto
evolucion_voto = {}
for categoria in ['Participacion', 'Decision']:
    fecha = []
    dias = []
    encuestadora = []
    votos = []
    pesos = []
    promedios = []
    sigmas = []
    for encuesta in bd_homologadas:
        if ~np.isnan(encuesta[categoria]):
            fecha.append(encuesta['Fecha'])
            dias.append(encuesta['Dias'])
            encuestadora.append(encuesta['Encuestadora'])
            votos.append(encuesta[categoria])
            pesos.append(encuesta['Peso'])
            if size(dias) > 1:
                promedios.append(np.dot(votos, pesos) / sum(pesos))
                sigmas.append(sqrt(mean((np.asarray(votos[1:]) - np.asarray(promedios))**2)))
                if sigmas[-1] < 0.001:
                    sigmas[-1] = std(config_file['participacion historica 1era'])
    evolucion_voto[categoria] = {'fecha': fecha[1:], 'dias': dias[1:], 'promedio': promedios, 'sigma': sigmas,
                                 'proyeccion': None}
for candidato in candidatos:
    fecha = []
    dias = []
    encuestadora = []
    votos = []
    pesos = []
    promedios = []
    stds = [0, ]
    sigmas = []
    ind = 0
    for encuesta in bd_homologadas:
        if ~np.isnan(encuesta[candidato]):
            fecha.append(encuesta['Fecha'])
            dias.append(encuesta['Dias'])
            encuestadora.append(encuesta['Encuestadora'])
            votos.append(encuesta[candidato])
            pesos.append(encuesta['Peso'])
            if size(dias) > 1:
                promedios.append(np.dot(votos, pesos) / sum(pesos))
                err_participacion = (promedios[-1]/evolucion_voto['Participacion']['promedio'][ind])*evolucion_voto['Participacion']['sigma'][ind]
                stds.append(sqrt(mean((np.asarray(votos[1:]) - np.asarray(promedios))**2) + encuesta['Asignaciones'][candidato]**2/12 + err_participacion**2))
                sigmas.append(np.dot(stds, pesos) / sum(pesos))
                ind += 1
    evolucion_voto[candidato] = {'fecha': fecha[1:], 'dias': dias[1:], 'promedio': promedios, 'sigma': sigmas}

candidatos = sorted(evolucion_voto, key=lambda k: evolucion_voto[k]['promedio'][-1], reverse=True)
candidatos.remove('Participacion')
candidatos.remove('Decision')


# Grafico con encuestas homologadas
fig = plt.figure()
ax = plt.subplot(111)
ind = 0
for candidato in candidatos:
    x = []
    y = []
    for encuesta in bd_homologadas:
        x.append(encuesta['Dias'])
        y.append(encuesta[candidato])
    ax.scatter(x, 100.0 * np.asarray(y), color=papeleta[candidatos[ind]], marker='.')
    ind += 1

z = []  # linea de tendencia lowess
proyecciones = [] # al dia de las elecciones
ind = 0
for candidato in candidatos:
    x = evolucion_voto[candidato]['dias']
    y = evolucion_voto[candidato]['promedio']
    err = evolucion_voto[candidato]['sigma']
    # next 6 lines: remove entries with same date
    rep = [i for i in range(len(x)) if not i == x.index(x[i])]
    rep = [i - 1 for i in rep]
    [x.pop(i) for i in sorted(rep, reverse=True)]
    [y.pop(i) for i in sorted(rep, reverse=True)]
    [err.pop(i) for i in sorted(rep, reverse=True)]
    [evolucion_voto[candidato]['fecha'].pop(i) for i in sorted(rep, reverse=True)]
    #ax.errorbar(x, 100.0 * np.asarray(y), 200.0 * np.asarray(err), color=papeleta[candidatos[ind]], fmt='o')
    #ax.scatter(x, 100.0 * np.asarray(y), color=papeleta[candidatos[ind]], marker='h', linewidths=5)
    # adev_evolucion_voto(x, y)
    # title(candidato)
    z.append(lowess(y, x, frac=3.0*tau/(x[-1]-x[0])))
    ax.plot(z[ind][:, 0], 100.0 * z[ind][:, 1], color=papeleta[candidatos[ind]], label=candidato, linewidth=2.5,
            zorder=len(candidatos)-ind)
    slopes = np.diff(z[ind][:, 1]) / np.diff(z[ind][:, 0])
    proyecciones.append(np.abs(z[ind][-1, 1] - slopes[-1] * x[-1]))
    ind += 1

proyecciones = proyecciones / sum(proyecciones)
for ind in range(size(candidatos)):
    evolucion_voto[candidatos[ind]]['proyeccion'] = proyecciones[ind]
    ax.plot((z[ind][-1, 0], 0), (100.0 * z[ind][-1, 1], 100.0 * proyecciones[ind]), '--', color=papeleta[candidatos[ind]])
    # ax.plot((z[ind][-1, 0], 0), (100.0 * z[ind][-1, 1], 100.0 * z[ind][-1, 1]), '--', color=papeleta[candidatos[ind]])

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.85])
ax.legend(loc='upper center', fontsize='x-small', bbox_to_anchor=(0.475, -0.15), ncol=5)
ax.axis('tight')
ax.set_xlim(1.02*min(x), 0)
y_lims = ax.get_ylim()
ax.set_ylim((0, y_lims[1]))
ax.set_xlabel(u'días a la elección')
ax.set_ylabel(u'% votos válidos')
ax.grid(color='k', alpha=0.25)
plt.show()
fig.savefig(config_file['promedios json file']+'.png', dpi=330)

with open(config_file['promedios json file']+'.json', 'w') as f:
    json.dump(evolucion_voto, f)
