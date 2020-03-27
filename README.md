# Natural-Questions [WIP]
Solving the problem of multi-hop open domain question answering on the Natural Questions dataset created at Google.

# Dependency list

The list of packages required for this project:

*	absl-py==0.9.0
*	astor==0.8.1
*	boto3==1.12.5
*	botocore==1.15.5
*	cachetools==4.0.0
*	certifi==2019.11.28
*	chardet==3.0.4
*	Click==7.0
*	cycler==0.10.0
*	docutils==0.15.2
*	filelock==3.0.12
*	Flask==1.1.1
*	Flask-Cors==3.0.8
*	gast==0.2.2
*	gevent==1.4.0
*	google-auth==1.11.2
*	google-auth-oauthlib==0.4.1
*	google-pasta==0.1.8
*	greenlet==0.4.15
*	grpcio==1.27.2
*	h5py==2.10.0
*	idna==2.9
*	itsdangerous==1.1.0
*	Jinja2==2.11.1
*	jmespath==0.9.4
*	joblib==0.14.1
*	Keras-Applications==1.0.8
*	Keras-Preprocessing==1.1.0
*	kiwisolver==1.1.0
*	Markdown==3.2.1
*	MarkupSafe==1.1.1
*	matplotlib==3.1.3
*	mock==4.0.1
*	more-itertools==8.2.0
*	numpy==1.18.1
*	oauthlib==3.1.0
*	opt-einsum==3.1.0
*	pandas==1.0.1
*	Pillow==7.0.0
*	protobuf==3.11.3
*	pyasn1==0.4.8
*	pyasn1-modules==0.2.8
*	pyparsing==2.4.6
*	python-dateutil==2.8.1
*	pytz==2019.3
*	regex==2020.2.20
*	requests==2.23.0
*	requests-oauthlib==1.3.0
*	rsa==4.0
*	s3transfer==0.3.3
*	sacremoses==0.0.38
*	scikit-learn==0.22.2
*	scipy==1.4.1
*	seaborn==0.10.0
*	sentencepiece==0.1.85
*	six==1.14.0
*	tensorboard==2.0.2
*	tensorflow==2.0.0
*	tensorflow-estimator==2.0.1
*	tensorflow-hub==0.7.0
*	termcolor==1.1.0
*	tokenizers==0.5.0
*	torch==1.4.0
*	torchvision==0.5.0
*	tornado==6.0.3
*	tqdm==4.43.0
*	transformers==2.5.0
*	urllib3==1.25.8
*	Werkzeug==1.0.0
*	wrapt==1.12.0


# Structure of the project


The Following is the structure of the project.
    
    examples/
    
    utils/
    
    train/

    ai/
        encoders/

    data/
        
    test/
    
    meetings/

    models/

    preprocessing/

    deploy/


# Creating a smaller sample

Due to the Natural Questions dataset being really huge, we create a smaller sample of the dataset which is easy to graps and work with. In order to create a smaller sample of the dataset, we follow the following procedure:
* First, we collect all the questions from the dataset.

* Second, we transform all the questions into their corresponding representations using a pre-trained text encoder model. In this case, we have used the fourth version of the **Universal Sentence Encoder** model built by **Google** as the encoder. This model provides an embedding given a sentence/text.
 
* Cluster the questions using **K-Means algorithm**. The clusters should probably contain similar questions based on some latent aspect.

* We consider these clusters as buckets and randomly sample a small subset of questions from each bucket.

* Finally, we aggregate the samples obtained from different buckets and write them to a file. 

Upon following the above procedure we obtained a smaller dataset with **89471** samples which is roughly **30%** of the total number of samples in the nq-train.jsonl dataset.

# GDrive Link

For accessing the dataset files and colab implementations, refer to the google drive folder given below.
https://drive.google.com/open?id=1z4bBWdxKygnIh_OGOR8juyOZviH-g4Ok

Note: The current implementation is under the folder name v3.
