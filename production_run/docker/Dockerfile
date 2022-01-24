FROM legacysurvey/legacypipe:DR9.6.5

ADD . /src

RUN pip install pybind11 proxmin peigen autograd

RUN git clone https://github.com/pmelchior/scarlet.git && cd scarlet && python setup.py install


