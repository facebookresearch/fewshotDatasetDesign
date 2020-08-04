from tensorboardX import SummaryWriter
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
from utils import *
from model import Model
from datasets import get_transform, get_dataset,dset_from_json
import torch.backends.cudnn as cudnn

class Trainer():
    def __init__(self, opt, dset=None):
        logdir = opt.logdir
        ###
        try:
            import submitit
            job_env = submitit.JobEnvironment()
            logdir = logdir.replace('%j', str(job_env.job_id))
            opt.logdir = logdir
        except:
            print('No job id found')
        ###
        if opt.ngpus > 1:
            opt.bSz = opt.bSz*opt.ngpus
            opt.n_workers = int(min(opt.n_workers*opt.ngpus, 20))
        self.opt = opt
        print(f'Training with opts: {opt}')
        
        self.writer = SummaryWriter(logdir)
        print(f'Log dir: {self.writer.log_dir}')
        self.writer.add_text('opts', str(opt), 0)
        
        # Fix seed
        if opt.seed: torch.manual_seed(opt.seed)
        
        # depending on the chosen architecture adapt training image size
        if '224' in opt.feat_arch:
            opt.iSz = 224
            print(f'Using iSz: {opt.iSz}')
        else:
            print(f'Continuing with iSz: {opt.iSz}')
        
        # construct train dataset or use provided one
        if dset is None:
            self.traindset = get_dataset(opt.dataset, classDset=True, iSz=opt.iSz)
        else:
            self.traindset = dset
            
        print(self.traindset)
        print(self.traindset.classes[0].samples[0])
        print('Train dataset class length histogram')
        print(np.histogram([len(c) for c in self.traindset.classes]))
        self.ttype = 'IN' if opt.benchmark == 'IN' else 'miniIN'
        self.traindset.transform = get_transform(self.ttype, phase='train', do_normalize=True, iSz=opt.iSz)
        print('Train transform: ', self.traindset.transform)
        # construct dataloader
        self.init_dataloader(self.traindset)

        # construct validation/test dataset
        self.get_val_test_sets()
        print('val dataset: ', self.valdset)
        print('test dataset: ', self.testdset)
        
        # verify image size
        assert opt.iSz in [224, 84], f' Got iSz: {opt.iSz}'
        
        # construct model
        self.model = Model(feat_arch=opt.feat_arch, nClasses=len(self.traindset.classes))
        if opt.ngpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(opt.ngpus))
            print('Using ')
        self.model.cuda()
        print(self.model)
        
        if opt.steps is None:
            opt.steps = get_steps(len(self.traindset), bSz=opt.bSz)
        print(f'Using steps: {opt.steps}')
        opt.max_iter = opt.steps[-1]
        
        # setup optimizer and scheduler
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd, nesterov=opt.nesterov)
        self.scheduler = MultiStepWarmupLR(self.optimizer, milestones=opt.steps, gamma=opt.gamma, warmup_steps=opt.warmup_steps)
        
        self.iteration = 0
        
        self.ims = torch.FloatTensor().cuda()
        self.targets = torch.LongTensor().cuda()
        self.best_5shot = 0
        self.best_ckpt_file = os.path.join(self.writer.log_dir,'best_checkpoint.pth')
        
        cudnn.benchmark = True
        print(f'Dataset size: {len(self.traindset)}, bSz: {opt.bSz}, steps: {opt.steps}, len dataloader {len(self.trainloader)}')
        

    def train(self):
        if self.opt.train_type == 'CC':
            return self.train_CosineClassifier()
        else:
            raise NotImplementedError(f'Unknown training type {self.opt.train_type}')

    def train_CosineClassifier(self):
        print('Start training CC !')
        while self.iteration < self.opt.max_iter:
            startTime = time.time()
            for i, (ims, targets) in enumerate(self.trainloader):
                self.ims.resize_(ims.size()).copy_(ims)
                self.targets.resize_(targets.size()).copy_(targets)

                self.model.train()
                self.optimizer.zero_grad()
                
                outputs = self.model(self.ims)
                
                loss = F.cross_entropy(outputs, self.targets)
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                lr = self.scheduler.get_lr()[0]

                if self.iteration % 50 == 0:
                    print(f'Iteration {self.iteration}: loss {loss:.3f}')
                    self.writer.add_scalar('Train_CC/lr', lr, self.iteration)
                    self.writer.add_scalar('Train_CC/loss', loss, self.iteration)
                    _, y_hat = outputs.max(1)
                    acc_val = torch.eq(y_hat, self.targets.squeeze()).float().mean()
                    self.writer.add_scalar('Train_CC/acc', acc_val, self.iteration)
                
                del loss, outputs # clean a bit
                
                if self.iteration %self.opt.eval_freq == 0:
                    # eval and plot
                    ret = self.eval_fewshot(self.valdset.sample_N_images(50000))# sample max validation images for a dataset
                    self.writer.add_scalar('Val_CC/1-shot', ret[1][0], self.iteration)
                    self.writer.add_scalar('Val_CC/5-shot', ret[5][0], self.iteration)
                     # if best checkpoint save it
                    if ret[5][0] > self.best_5shot:
                        self.best_5shot = ret[5][0]
                        state_dict = self.model.state_dict() if self.opt.ngpus == 1 else self.model.module.state_dict()
                        obj = {
                            'model':state_dict,
                            'iteration':self.iteration,
                            'best_5shot': self.best_5shot,
                            'opt':self.opt,
                        }
                        print(f'Saving best model at iteration: {self.iteration}, {self.best_5shot}')
                        torch.save(obj, self.best_ckpt_file)
                        torch.save(ret, os.path.join(self.writer.log_dir, 'bestval_result.pth'))
                
                self.iteration += 1
                if self.iteration >= self.opt.max_iter:
                    break
            epochTime = time.time()- startTime
            eta = epochTime*(self.opt.max_iter-self.iteration)/len(self.trainloader)
            print(f'ETA: {eta/3600:0.0f}h {(eta%3600)/60:0.0f}min')
        # eval and save
        # load best checkpoint
        print('Loading best model')
        x = torch.load(self.best_ckpt_file)
        if self.opt.ngpus == 1:
            self.model.load_state_dict(x['model'])
        else:
            self.model.module.load_state_dict(x['model'])
            
        ret = self.eval_fewshot(self.testdset, nEpisodes=self.opt.nEpisodes, test=True) # TODO
        torch.save(ret, os.path.join(self.writer.log_dir,'result.pth'))
        if self.opt.delete_ckpt:
            os.remove(self.best_ckpt_file)
        self.writer.add_scalar('Test_CC/1-shot', ret[1][0], self.iteration)
        self.writer.add_scalar('Test_CC/5-shot', ret[5][0], self.iteration)
        return ret
        
    def eval_fewshot(self, whole_test_set, kshots=[1,5], seed_episode=True, usemean=True, nEpisodes=1000, test=False):
        torch.cuda.empty_cache()
        start_time = time.time()
        classifier_arch = 'cosine'
        # precompute features of the whole set
        self.model.eval()
        model = self.model if self.opt.ngpus == 1 else self.model.module
        
        # eval few-shot
        results = {}
        ret = {}
        for k in kshots:
            results[k] = []
        maxk = max(kshots)
        nNovel = self.nNovel; nTest = self.nTest; topk = self.topk
        
        set_json = self.opt.test_json if test else self.opt.val_json
        if set_json:
            if set_json == 'miniIN1k_nim100_seed0_closest':
                classDict = torch.load('data/test_benchmarks/miniIN1k_4_closest_inds_of_each.pth')
                print('Closest class benchmark')
            elif set_json == 'miniIN1k_nim100_seed0_farthest':
                classDict = torch.load('data/test_benchmarks/miniIN1k_4_farthest_inds_of_each.pth')
                print('Farthest class benchmark')
            elif set_json == 'miniIN1k_nim100_seed0':
                classDict = {}
                print('Random class benchmark')
            elif 'diverse' in set_json:
                print('Using most diverse classes')
                if set_json == 'miniIN1k_most_diverse':
                    classInds = torch.load('data/test_benchmarks/100_high_diversity_inds.pth')
                elif set_json == 'miniIN1k_least_diverse':
                    classInds = torch.load('data/test_benchmarks/100_low_diversity_inds.pth')
                whole_test_set = whole_test_set.sample_n_classes(class_inds=classInds)
                print('Sub-sampled dataset ', whole_test_set)
        print(f'Precomputing features of whole set with: {len(whole_test_set)} images, test: {str(test)}')
        testloader = DataLoader(whole_test_set, batch_size=self.opt.bSz, shuffle=False, num_workers=20)
        features, targets = compute_features(model.feat, testloader, verbose=False)
        print('Done. Got features of size: {}'.format(features.size()))

        for episode in range(nEpisodes):
            seed = episode if seed_episode else None
            g = get_generator(seed)
            N = len(whole_test_set)
            if set_json and 'miniIN1k_nim100_seed0' in set_json:
                if len(classDict):
                    c0ind = torch.randperm(1000)[0].item()
                    class_inds = torch.LongTensor([c0ind]+classDict[c0ind])
                else:
                    class_inds = torch.randperm(1000)[:5]
            else:
                non_empty_class_inds = [i for i,c in enumerate(whole_test_set.classes) if len(c)]
                
                class_inds = torch.randperm(len(non_empty_class_inds), generator=g)[:nNovel]
                class_inds = torch.LongTensor(non_empty_class_inds)[class_inds]
                
            global_inds = []
            for class_ind in class_inds:
                class_size = len(whole_test_set.classes[class_ind])
                sampled_inds = (torch.randperm(class_size, generator=g)[:maxk+nTest]).tolist()
                global_inds.extend([whole_test_set.cum_len[class_ind] + ind for ind in sampled_inds])
            global_inds = torch.LongTensor(global_inds).view(nNovel, maxk+nTest)
            targets = torch.LongTensor(range(nNovel)).repeat(maxk+nTest,1).transpose(0,1)

            exemplar_inds = global_inds[:,:maxk].contiguous().view(-1)
            test_inds = global_inds[:,maxk:].contiguous().view(-1)

            trainfeatures_maxk, traintargets_maxk = features[exemplar_inds], targets[:,:maxk].contiguous().view(-1)
            testfeatures, testtargets = features[test_inds], targets[:,maxk:].contiguous().view(-1)

            for k in kshots:
                inds = torch.cat([torch.where(traintargets_maxk==i)[0][0:k] for i in range(nNovel)])
                trainfeatures = trainfeatures_maxk[inds]
                traintargets = traintargets_maxk[inds]

                if classifier_arch == 'cosine':
                    testfeatures = testfeatures.cuda()
                    trainfeatures = trainfeatures.cuda()
                    traintargets = traintargets.cuda()

                    # Cosine Similarity with average feature
                    cls_score = model.fc(testfeatures, features_train=trainfeatures, labels_train=traintargets)

                    # eval on val_noveldset by computing NN to features in train_noveldset
                    accs = accuracy(cls_score.cpu(), torch.LongTensor(testtargets), topk=(topk,))
                else:
                    x = F.normalize(testfeatures, p=2, dim=testfeatures.dim()-1)# len(val_noveldset)-->bSz,fSz
                    y = F.normalize(trainfeatures, p=2, dim=trainfeatures.dim()-1)# len(train_noveldset)-->nclasses*nExemplar_per_class, fSz
                    dist = torch.mm(x,y.t()) # size: len(val_noveldset)-->bSz, allExemplars
                    vals, inds = traintargets.sort()
                    dist = dist[:,inds].view(dist.size(0),nNovel, k)
                    if usemean:
                        dist = dist.mean(2)
                    else:
                        dist = dist.min(2)[0]# take min distance over nExemplars of each class
                    # eval on val_noveldset by computing NN to features in train_noveldset
                    accs = accuracy(dist, torch.LongTensor(testtargets), topk=(topk,))
                results[k].append(accs)
                ret[k] = get_mean_ci(np.array(results[k])[:,0])
            
            if test:
                self.writer.add_scalar('Test_CC/1-shot', ret[1][0], episode)
                self.writer.add_scalar('Test_CC/5-shot', ret[5][0], episode)
            if episode % 1000 == 0:
                print('----------')
                print(f'Episode: {episode}/{nEpisodes}')
                for k in kshots:
                    print(' {}-shot top-{} acc: {:0.2f}±{:0.2f}'.format(k, topk, *ret[k]))
                print('----------')
        print('Final')
        print('----------')
        for k in kshots:
            print('  {}-shot top-{} acc: {:0.2f}±{:0.2f}'.format(k, topk, *ret[k]))
        print('----------')
        s = time.time()-start_time
        print(f'Took: {s:0.2f}s')
        return ret
        
    def init_dataloader(self, dset):
        class_sample_count = [len(c) for c in dset.classes]
        weights = 1 / torch.Tensor(class_sample_count)
        weights[~((weights + 1) != weights)]= 0
        weight_per_sample = [0]*len(dset)
        for i in range(len(dset)):
            c, cind = dset.index_to_sample_d[i]
            weight_per_sample[i] = weights[c]
        self.trainsampler = sampler.WeightedRandomSampler(weight_per_sample, len(dset))
        self.trainloader = DataLoader(dset, batch_size=self.opt.bSz, pin_memory=True, num_workers=self.opt.n_workers, sampler=self.trainsampler, drop_last=True)
        
    def get_val_test_sets(self):
        if self.opt.benchmark in ['miniIN','cub', 'tieredIN','flower'] or 'miniIN1k' in self.opt.benchmark:
            if 'miniIN1k' in self.opt.benchmark:
                self.valdset = get_dataset('miniIN', split='val', iSz=self.opt.iSz)
            else:
                self.valdset = get_dataset(self.opt.benchmark, split='val', iSz=self.opt.iSz)
                
            if self.opt.test_json:
                self.testdset = dset_from_json('data/test_benchmarks/miniIN1k_nim100_seed0.json')
            else:
                self.testdset = get_dataset(self.opt.benchmark, split='test', iSz=self.opt.iSz)
            self.nNovel = 5
            self.nTest = 15 # per class
            self.topk = 1
        elif self.opt.benchmark == 'IN':
            self.valdset = get_dataset(self.opt.benchmark, split='val')
            self.testdset = get_dataset(self.opt.benchmark, split='test')
            self.nNovel = 250
            self.nTest = 6 # per class
            self.topk = 5
        else:
            raise NotImplementedError('Need to implement eval set for IN')
        
        transform = get_transform(self.ttype, phase='test', do_normalize=True, iSz=self.opt.iSz)
        self.valdset.transform = transform
        self.testdset.transform = transform


class EpisodicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dset, nway=5, nsupport=1, nquery=10, seed=None):
        self.nepisodes = 1+len(dset)//(nway*(nquery+nsupport))
        self.dset = dset
        self.nway = nway
        self.nsupport = nsupport
        self.nquery = nquery
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = []
        for i in range(self.nepisodes):
            # sample class indices
            seed_i = None if self.seed is None else self.seed+i+self.epoch
            gen = get_generator(seed_i)
            class_inds = torch.randperm(len(self.dset.classes), generator=gen)[:self.nway].tolist()
            for ind in class_inds:
                c = self.dset.classes[ind]
                cinds = torch.randperm(len(c), generator=gen)[:self.nsupport+self.nquery]
                start = self.dset.cum_len[ind]; end = self.dset.cum_len[ind+1]
                cinds = torch.LongTensor(range(start, end))[cinds]
                indices.append(cinds)
        indices = torch.cat(indices)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.nepisodes