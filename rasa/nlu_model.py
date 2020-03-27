from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter



def train_nlu(data,config,model_dir):
	train_data = load_data(data)
	trainer = Trainer(RasaNLUConfig(config))
	trainer.train(train_data)
	model_directory = trainer.persist(model_dir,fixed_model_name='meathernlu')

def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/weathernlu',RasaNLUConfig('config_spacy.json'))
	print(interpreter.parse("I am plaaning my holiday to Beijing, I wonder what is the weather out there"))




	
if __name__ == '__main__':
	# train_nlu('./data/data.json','config_spacy.json','./model/nlu')
	run_nlu()
