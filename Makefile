VPATH = src:headers
CC = mpicc
CFLAGS = -g -Wall
CPPFLAGS = -Iheaders/

all: trainmaster_mpi testsync_mpi ridgeline_md

trainmaster_mpi: trainmaster_mpi.c hmm_mpi.h matrix.o cluster_mpi.o prob.o estimate_mpi.o modelio.o modal.o utils_mpi.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -lm -o $@

testsync_mpi: testsync_mpi.c hmm_mpi.h matrix.o cluster_mpi.o prob.o estimate_mpi.o modelio.o modal.o utils_mpi.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -lm -o $@

ridgeline_md: ridgeline_md.c hmm_mpi.h agglomerate.h matrix.o prob.o modal.o modelio.o agglomerate.o estimate_mpi.o cluster_mpi.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -lm -o $@

matrix.o: matrix.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

utils_mpi.o: utils_mpi.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

cluster_mpi.o: cluster_mpi.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

prob.o: prob.c hmm_mpi.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

estimate_mpi.o: estimate_mpi.c hmm_mpi.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

modelio.o: modelio.c hmm_mpi.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

modal.o: modal.c hmm_mpi.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

agglomerate.o: agglomerate.c agglomerate.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm *.o
