import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import time


config_file = json.load(open('presidente/config_presidente.json', 'rb'))


def graficar_intervalos(sim_results, candidatos, papeleta, intervalos_output_file):
    N_candidatos = len(candidatos)
    fig_i, ax_i = plt.subplots()
    for ind in range(N_candidatos):
        media = sim_results[ind]['media']
        lim_inferior = sim_results[ind]['extremo inferior']
        rango = sim_results[ind]['extremo superior'] - lim_inferior
        ax_i.broken_barh([(lim_inferior,rango)],(10*(N_candidatos-ind),9),facecolors = papeleta[candidatos[ind]])
        ax_i.broken_barh([(media, 0.25)], (10 * (N_candidatos - ind) + 1, 7), facecolors='#ffffff')
    box = ax_i.get_position()
    ax_i.set_position([box.x0 + box.width * 0.2, box.y0, box.width * 0.85, box.height])
    ax_i.set_yticks(linspace(15,N_candidatos*10+5,N_candidatos))
    candidatos.reverse()
    ax_i.set_yticklabels(candidatos)
    ax_i.set_xlabel(u'% votos válidos')
    x_lims = ax_i.get_xlim()
    ax_i.set_xlim((0,x_lims[1]))
    ax_i.grid(axis='x', linestyle='--')
    ax_i.set_axisbelow(True)
    ax_i.set_title('Intervalos de Confianza')
    plt.show()
    savefig(intervalos_output_file + '.png', dpi=330)


t = time.time()

promedios_file = config_file['promedios json file'] # archivo con evolucion de promedios ponderados
output_fname = config_file['probabilidades output fname']
histograma_output_file = config_file['histograma fname']
intervalos_output_file = config_file['intervalos fname']
output_txt_file = open(output_fname + '.txt', 'w')
output_csv_file = output_fname + '.csv'

evolucion_voto = json.load(open(promedios_file + '.json', 'rb'))
evolucion_voto.pop('Participacion')
evolucion_voto.pop('Decision')
candidatos = sorted(evolucion_voto, key=lambda k: evolucion_voto[k]['proyeccion'], reverse=True)
# candidatos = sorted(evolucion_voto, key=lambda k: evolucion_voto[k]['promedio'][-1], reverse=True)

papeleta = config_file['papeleta']

N = 50000
alphas = {}
betas = {}
FDP = {}

for candidato in candidatos:
    mu = evolucion_voto[candidato]['proyeccion']
    # mu = evolucion_voto[candidato]['promedio'][-1]
    sigma = evolucion_voto[candidato]['sigma'][-1]
    alphas[candidato] = mu*(mu*(1-mu)/sigma**2 - 1)
    betas[candidato] = (1 - mu)*(mu*(1-mu)/sigma**2 - 1)
    FDP[candidato] = np.random.beta(alphas[candidato], betas[candidato], size=N)

try:
    del trayectorias_todas
except:
    print('This is the first time you run this script. Array with all trajectories has not been previously defined.')

#suma = []
trayectorias_todas = np.ndarray((1,np.size(candidatos)))
for n in range (0,N):
    trayect = []
    for candidato in candidatos:
        trayect.append(FDP[candidato][n])
    trayect = np.asarray(trayect)
    trayect[1:] = (1.0 - trayect[0])/sum(trayect[1:])*trayect[1:]
    #suma.append(sum(trayect))
    trayectorias_norm = trayect
    trayectorias_todas = np.concatenate((trayectorias_todas,np.asarray([trayectorias_norm])))
trayectorias_todas = np.delete(trayectorias_todas,(0), axis=0)
#plt.figure()
#plt.hist(suma,100)
#plt.show()

# Histograms
sim_results = [] # Lista de diccionarios con resultados por candidato
output_txt_file.write('%%%%%% Medias, Medianas, Desviaciones Estandar e Intervalos de Confianza %%%%%%')
fig1, axarr = plt.subplots(np.size(candidatos), sharex=True, sharey=True)
for n in range (np.size(candidatos)):
    axarr[n].hist(trayectorias_todas[:,n],50,color=papeleta[candidatos[n]])
    axarr[n].grid()
    media = 100*mean(trayectorias_todas[:,n])
    mediana = 100*median(trayectorias_todas[:,n])
    desviacion = 100*std(trayectorias_todas[:,n])
    lim_inferior = 100*percentile(trayectorias_todas[:,n],2.5)
    lim_superior = 100*percentile(trayectorias_todas[:,n],97.5)
    rango_95 = lim_superior - lim_inferior
    output_txt_file.write('\n' + candidatos[n] + ':   ' + str(round(media,1)) + '   ' +str(round(mediana,1)) + '   ' +  str(round(desviacion,1)) + '   [' + str(round(lim_inferior,1)) + ', ' + str(round(lim_superior,1)) + '] (' + str(round(rango_95,1)) + ')')
    sim_results.append({'candidato': candidatos[n], 'media': media, 'mediana': mediana, 'sigma': desviacion, 'extremo inferior': lim_inferior, 'extremo superior': lim_superior})
fig1.subplots_adjust(hspace=0)
plt.xlim(0,0.7)
plt.show()
savefig(histograma_output_file + '.eps')

posicion = np.argsort(-trayectorias_todas, axis=1) # Ordena indice de candidatos de mayor a menor
trayectorias_ordenadas = -np.sort(-trayectorias_todas, axis=1)  # Ordena trayectorias de mayor a menor

# Candidatos que ganan la primera vuelta
gana_primera = np.unique(posicion[:,0], return_counts=1)
output_txt_file.write('\n\n%%%%%% Probabilidad de Ganar la Primera Vuelta %%%%%%')
for x in range(size(gana_primera[0])):
    prob_frac = gana_primera[1][x]/np.float(N)
    sim_results[x]['probabilidad de ganar'] = prob_frac
    output_txt_file.write('\n' + candidatos[gana_primera[0][x]] + ': %.1f' %(prob_frac*100.0) + '%')

# Probabilidad de Ganar en una sola vuelta
mas_de_50 = where(trayectorias_ordenadas[:,0] > 0.5)
mas_de_50 = set(mas_de_50[0]) # Indices donde el primer lugar obtiene mas del 50%

presi_deuna = []
output_txt_file.write('\n\n%%%%%% Probabilidad de Ganar Directamente %%%%%%')
for x in range(size(gana_primera[0])):
    primero_ind = where(posicion[:,0] == gana_primera[0][x])
    primero_ind = set(primero_ind[0])
    presi_50 = primero_ind.intersection(mas_de_50)
    presi_deuna.append(list(presi_50))
    output_txt_file.write('\n' + candidatos[gana_primera[0][x]] + ': %.1f' %(size(presi_deuna[x])*100.0/N) + '%')

output_txt_file.write('\n\n%%%%%% Habra segunda vuelta? %%%%%%')
no_segunda = mas_de_50
trayect_todas_ind = set(np.linspace(0,N-1,N))
si_segunda = trayect_todas_ind.difference(no_segunda)
output_txt_file.write('\n' + 'Si: %.1f' %(size(list(si_segunda))*100.0/N) + '%')

output_txt_file.write('\n\n%%%%%% Quien pasa a segunda vuelta? %%%%%%')
candidatos_segunda_arr = np.asarray(list(si_segunda))
candidatos_segunda_ind = posicion[candidatos_segunda_arr.astype(np.int),0:2]
candidatos_segunda_list = []
for x in candidatos_segunda_ind:
    candidatos_segunda_list.append(candidatos[x[0]] + ' vs. ' + candidatos[x[1]])

binomios_segunda = np.unique(candidatos_segunda_list, return_counts=1)
for x in range(size(binomios_segunda[0])):
    output_txt_file.write('\n' + binomios_segunda[0][x] + ': %.1f' %(binomios_segunda[1][x]*100.0/N) + '%')

elapsed = time.time() - t
output_txt_file.write('\n\nElapsed time: %.1f seconds.' %elapsed)

output_txt_file.close()

# Figure with confidence intervals
graficar_intervalos(sim_results, candidatos, papeleta, intervalos_output_file)

# Archivo csv
keys = sim_results[0].keys()
with open(output_csv_file, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(sim_results)
