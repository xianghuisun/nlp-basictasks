import importlib
from .transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup,get_constant_schedule,get_linear_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup,get_constant_schedule_with_warmup,get_cosine_schedule_with_warmup
from .transformers.optimization import AdamW,Adafactor

def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int ):
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))

def get_optimizer(model,optimizer_type,weight_decay,optimizer_params):
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if optimizer_type=='AdamW':
        optimizer=AdamW(optimizer_grouped_parameters, **optimizer_params)
    elif optimizer_type=='Adafactor':
        optimizer=Adafactor(optimizer_grouped_parameters, **optimizer_params)
    else:
        raise Exception("Unknown optimizer type")
    return optimizer



def fullname(o):
  """
  Gives a full name (package_name.class_name) for a class / object in Python. Will
  be used to load the correct classes from JSON files
  """

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
      return o.__class__.__name__  # Avoid reporting __builtin__
  else:
      return module + '.' + o.__class__.__name__

def import_from_string(dotted_path):
    '''
    如果输入是torch.nn.modules.activation.Tanh，那么module_path=torch.nn.modules.activation，class_name=Tanh
    对应的module是<module 'torch.nn.modules.activation' from '/root/miniconda3/envs/nlp_sr/lib/python3.6/site-packages/torch/nn/modules/activation.py'>
    返回的是torch.nn.modules.activation.Tanh

    如果输入是nlp_basictask.modules.transformers.BertModel
    对应的module是<module 'nlp_basictask.modules.transformers' from '/root/NLP_warehouse/nlp_basictask/modules/transformers/__init__.py'>
    返回的是nlp_basictask.modules.transformers.BertModel

    也就是说根据传进来的string返回对应的Model
    '''
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)
    
    try:
        module = importlib.import_module(dotted_path)
    except:
        module = importlib.import_module(module_path)
    
    try:
        return getattr(module,class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)