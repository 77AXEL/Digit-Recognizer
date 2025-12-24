F_TARGET = module
F_SRC    = module.f90

CPP_DLL  = optimized.dll
CPP_SRC  = optimized.cpp

.PHONY: all clean

all: fortran cpp

fortran: $(F_SRC)
	f2py -m $(F_TARGET) -c $(F_SRC) --fcompiler=gnu95 --opt="-O3 -march=native -funroll-loops -ffast-math"
	@for %%f in ($(F_TARGET)\.libs\*.dll) do copy /Y "%%f" .
	@rmdir /S /Q $(F_TARGET)

cpp: $(CPP_SRC)
	g++ -O3 -march=native -ffast-math -shared -o $(CPP_DLL) $(CPP_SRC)

clean:
	del /Q *.pyd *.o *.mod *.dll 2>NUL || true
