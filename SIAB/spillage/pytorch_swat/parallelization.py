import multiprocessing
import sys
def parallelization(nprocs: int, func: callable):

    if nprocs > 1:
        print("Parallelization is enabled, redirect stdout to files indiced by tasks.")
    ncores = multiprocessing.cpu_count()
    if nprocs > ncores:
        print("Warning: nprocs is larger than the number of cores, set nprocs to the number of cores.")
        nprocs = ncores

    procs = []
    for i in range(nprocs):
        sys.stdout = open("task_%d.log"%i, "w")
        procs.append(multiprocessing.Process(target=func, args=(i,)))
        procs[i].start()
    
    for i in range(nprocs):
        procs[i].join()

    if nprocs > 1:
        print("Parallelization is finished.")
    sys.stdout = sys.__stdout__
    print("All tasks are finished.")

