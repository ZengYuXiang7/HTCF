from modules.models.baselines.CSMF import CSMF
from modules.models.baselines.GATCF import GATCF
from modules.models.baselines.GraphMF import GraphMF
from modules.models.baselines.NeuCF import NeuCF
from modules.models.model import CF


def get_model(datasets, args):
    user_num, serv_num = datasets.data.shape
    # print(user_num, serv_num)
    if args.model == 'CF':
        return CF(user_num, serv_num, args)
    elif args.model == 'NeuCF':
        return NeuCF(user_num, serv_num, args)
    elif args.model == 'CSMF':
        return CSMF(user_num, serv_num, args)
    elif args.model == 'GraphMF':
        return GraphMF(user_num, serv_num, args)
    elif args.model == 'GATCF':
        return GATCF(user_num, serv_num, args)
    else:
        raise NotImplementedError
