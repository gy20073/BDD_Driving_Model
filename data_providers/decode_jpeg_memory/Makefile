CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

TF_INC = `python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"`

clean:
	rm decode_jpeg_batch.so

turbo:
	wget 'https://sourceforge.net/projects/libjpeg-turbo/files/1.5.1/libjpeg-turbo-official_1.5.1_amd64.deb/download' -O libjpeg-turbo_1.5.1_amd64.deb
	sudo dpkg -i libjpeg-turbo_1.5.1_amd64.deb
	sudo ln -s /opt/libjpeg-turbo/lib64/libturbojpeg.so /usr/lib/x86_64-linux-gnu/
	sudo ln -s /opt/libjpeg-turbo/lib64/libturbojpeg.so.0 /usr/lib/x86_64-linux-gnu
	sudo ln -s /opt/libjpeg-turbo/lib64/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu
	rm libjpeg-turbo_1.5.1_amd64.deb

tf:
	g++ -std=c++11 -shared decode_jpeg_batch.cpp -o decode_jpeg_batch.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lturbojpeg

all: turbo tf
