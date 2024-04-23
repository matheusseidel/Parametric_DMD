
import warnings

warnings.filterwarnings("ignore")

from pydmd import ParametricDMD, DMD, HankelDMD
from ezyrb import POD, RBF
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from pydmd.plotter import plot_eigs
#import h5py
import meshio
import os
import datetime
import time

start_time = datetime.datetime.now()

# -------- Parameters --------  

ti = 1                     # Initial timestep read
tf = 201                    # Final timestep read

num_nodes = 164343         # 0 a 164342
num_time_steps = tf - ti

training_params = np.array([100, 150, 250, 300, 350, 400, 450, 500, 600, 700])
testing_params = np.array([300, 301, 305])

rank_svd = 100
rank_pod = 100

save_vtk = 1

# -------- Reading Data --------  

data = []

for p in training_params:

    current_t = 0

    snapshots_p = np.zeros((num_nodes, num_time_steps))
    print(f'Reading data for Re={p}')

    for t in range(ti, tf):
        mesh_pressure_t = meshio.read(f'E:/Cilindro_{p}-vtk/Data_{t}.vtk')
        snapshots = mesh_pressure_t.point_data['pressure']
        print(f'Re={p}: Reading time step number {t}')
        snapshots_p[:, current_t] = snapshots[:, 0]
        current_t = current_t + 1
    
    print()

    data.append(snapshots_p)

training_snapshots = np.stack([data[p] for p in range(0, len(training_params))])
print('Training snapshots shape: ', training_snapshots.shape)
print()
print('Training parameters ', training_params)
print()

# -------- Create output log file -------- 

time = datetime.datetime.now()
time_title = str(time).replace(":",".")

print('Writing output log file')
print()

with open(f'E:/Parametric_DMD_output-{time_title[0:19]}.txt', 'w') as txt_log:
    text_time = str(time)
    txt_log.write('Parametric DMD: Output log ' + text_time + '\n')
    txt_log.write('\n')
    txt_ti = str(ti)
    txt_tf = str(tf)
    txt_log.write('Initial time step read: '+ txt_ti + '\n')
    txt_log.write('Final time step read: '+ txt_tf + '\n')
    txt_log.write('\n')
    txt_log.write('Parameter considered: Reynolds Number (Re)' + '\n')
    txt_training_params = str(training_params)
    txt_testing_params = str(testing_params)
    txt_log.write('Training parameters' + txt_training_params + '\n')
    txt_log.write('Testing parameters' + txt_testing_params + '\n')
    txt_log.write('\n')
    txt_rank_svd = str(rank_svd)
    txt_rank_pod = str(rank_pod)
    txt_log.write('SVD Rank: ' + txt_rank_svd + '\n')
    txt_log.write('POD Rank: ' + txt_rank_pod + '\n')
    txt_log.write('\n')

# -------- Monolithic variant - Training --------  

print('Running DMD')
dmd = DMD(svd_rank=rank_svd, tlsq_rank=rank_svd, exact=True, opt=True)

print('Running POD')
rom = POD(rank=rank_pod)

print('Running interpolator')
interpolator = RBF()

print('Running parametric DMD')
pdmd_monolithic = ParametricDMD(dmd, rom, interpolator)
pdmd_monolithic.fit(training_snapshots, training_params)

# -------- Testing new parameters --------

print('Testing new parameters')
pdmd_monolithic.dmd_time["t0"] = ti - 1 
pdmd_monolithic.dmd_time["tend"] = tf - 2
pdmd_monolithic.parameters = testing_params

result = pdmd_monolithic.reconstructed_data.real
print('Resulting snapshots shape: ', result.shape)
print()
print('Testing parameters ', testing_params)
print()
print()

# -------- Analyzing results --------

original_300 = training_snapshots[3, :, :]

current_t = 0
original_301 = np.zeros((num_nodes, num_time_steps))
print(f'Reading original data for Re=301')
print()

for t in range(ti, tf):
    mesh_pressure_t = meshio.read(f'E:/Cilindro_301-vtk/Data_{t}.vtk')
    snapshots = mesh_pressure_t.point_data['pressure']
    original_301[:, current_t] = snapshots[:, 0]
    current_t = current_t + 1

current_t = 0
original_305 = np.zeros((num_nodes, num_time_steps))
print(f'Reading original data for Re=305')
print()

for t in range(ti, tf):
    mesh_pressure_t = meshio.read(f'E:/Cilindro_305-vtk/Data_{t}.vtk')
    snapshots = mesh_pressure_t.point_data['pressure']
    original_305[:, current_t] = snapshots[:, 0]
    current_t = current_t + 1

approximation_300 = result[0, :, :]
approximation_301 = result[1, :, :]
approximation_305 = result[2, :, :]

print('Error for Re=300')
E300 = np.linalg.norm(original_300 - approximation_300, 'fro')/np.linalg.norm(original_300, 'fro')
print(E300)
print()

print('Error for Re=301')
E301 = np.linalg.norm(original_301 - approximation_301, 'fro')/np.linalg.norm(original_301, 'fro')
print(E301)
print()

print('Error for Re=305')
E305 = np.linalg.norm(original_305 - approximation_305, 'fro')/np.linalg.norm(original_305, 'fro')
print(E305)
print()

print('Writing output log file')
print()

with open(f'E:/Parametric_DMD_output-{time_title[0:19]}.txt', 'a') as txt_log:
    txt_training_snapshots_shape = str(training_snapshots.shape)
    txt_result_shape = str(result.shape)
    txt_log.write('Traning snaphots shape: ' + txt_training_snapshots_shape + '\n')
    txt_log.write('Testing snaphots shape: ' + txt_result_shape + '\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('Comparing results for Re=300' + '\n')
    txt_log.write('\n')
    txt_log.write('Original data for Re=300' + '\n')
    txt_original_300_shape = str(original_300.shape)
    txt_original_300 = str(original_300)
    txt_log.write('Shape:' + txt_original_300_shape + '\n')
    txt_log.write(txt_original_300 + '\n')
    txt_log.write('\n')
    txt_log.write('Parametric approximation for Re=300' + '\n')
    txt_approximation_300_shape = str(approximation_300.shape)
    txt_approximation_300 = str(approximation_300)
    txt_log.write('Shape:' + txt_approximation_300_shape + '\n')
    txt_log.write(txt_approximation_300 + '\n')
    txt_log.write('\n')
    txt_E300 = str(E300)
    txt_log.write('Error for Re=300: ' + txt_E300 + '\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('Comparing results for Re=301' + '\n')
    txt_log.write('\n')
    txt_log.write('Original data for Re=301' + '\n')
    txt_original_301_shape = str(original_301.shape)
    txt_original_301 = str(original_301)
    txt_log.write('Shape:' + txt_original_301_shape + '\n')
    txt_log.write(txt_original_301 + '\n')
    txt_log.write('\n')
    txt_log.write('Parametric approximation for Re=301' + '\n')
    txt_approximation_301_shape = str(approximation_301.shape)
    txt_approximation_301 = str(approximation_301)
    txt_log.write('Shape:' + txt_approximation_301_shape + '\n')
    txt_log.write(txt_approximation_301 + '\n')
    txt_log.write('\n')
    txt_E301 = str(E301)
    txt_log.write('Error for Re=301: ' + txt_E301 + '\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('Comparing results for Re=305' + '\n')
    txt_log.write('\n')
    txt_log.write('Original data for Re=305' + '\n')
    txt_original_305_shape = str(original_305.shape)
    txt_original_305 = str(original_305)
    txt_log.write('Shape:' + txt_original_305_shape + '\n')
    txt_log.write(txt_original_305 + '\n')
    txt_log.write('\n')
    txt_log.write('Parametric approximation for Re=305' + '\n')
    txt_approximation_305_shape = str(approximation_305.shape)
    txt_approximation_305 = str(approximation_305)
    txt_log.write('Shape:' + txt_approximation_305_shape + '\n')
    txt_log.write(txt_approximation_305 + '\n')
    txt_log.write('\n')
    txt_E305 = str(E305)
    txt_log.write('Error for Re=305: ' + txt_E305 + '\n')
    txt_log.write('\n')
    txt_log.write('\n')
    txt_log.write('\n')

end_time = datetime.datetime.now()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 

with open(f'E:/Parametric_DMD_output-{time_title[0:19]}.txt', 'a') as txt_log:
    txt_elapsed_time = str(elapsed_time)
    txt_log.write('Elapsed time: ' + txt_elapsed_time + '\n')


# -------- Saving vtk results --------

if save_vtk == 1:
    mesh_vtk = meshio.read('E:/Cilindro_100-vtk/Data_0.vtk')
    pressure = mesh_vtk.point_data['pressure']

    print('Writing results in vtk')
    print()
    i = 0

    for Re in testing_params:

        os.mkdir(f'E:/Parametric_DMD_output-{time_title[0:19]}_Re={Re}')

        for t in range(0, tf-ti, 5):
            print(f'Writing Re={Re} timestep {t}')
            step = result[i, :, t]
            mesh_vtk.point_data['pressure'] = step
            mesh_vtk.write(f'E:/Parametric_DMD_output-{time_title[0:19]}_Re={Re}/DMD_timestep_{t}.vtk')

        i = i + 1

