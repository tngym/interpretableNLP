FROM kaggle/python:latest

RUN pip install streamlit

RUN apt-get install -y build-essential swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
RUN pip install auto-sklearn
