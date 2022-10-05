F2PY = f2py3

all: smooth_gauss

.PHONY: all clean

smooth_gauss:
	$(F2PY) -c -m $@ $@.f

clean:
	rm -rf *.so
