#imports
import MCTS import MCTS
import torch
import random
import utils_mcts import ReplayBuffer, PathsBuffer, get_states_emb
from problem_mcts import GraphProblem, generate_erdos_renyi_problems, generate_regular_problems
from network_mcts import Agent

NUM_ITERS = 100
NUM_EPISODES = 100
PATH_LENGTH = 100
BATCH_SIZE = 32
NUM_MCSIMS = 300
NUM_UPDATES = 10
TEMP_THRESHOLD = 100
NUM_VERTICES = 10
CPUCT = 1

#initialize problem
problem_maker = generate_regular_problems(num_vertices=10, degree=3)
problem = next(problem_maker)
agent = Agent(hid_size=256, lstm_size=128, gcn_size=256, vertex_emb_size=64, num_vertices=NUM_VERTICES)
graph_emb = agent.embed_graph(problem.edges, **kwargs)

path_buffer = PathsBuffer()
train_buffer = ReplayBuffer()

for iteration in range(NUM_ITERS):
    for episode in range(NUM_EPISODES):
        
        mcts = MCTS(problem=problem, agent=agent, graph_emb=graph_emb, numMCTSSims=NUM_MCSIMS, cpuct=CPUCT, path_length=PATH_LENGTH)

        trainExamples = []
        path = game.get_state()
        episodeStep = 0
        while episodeStep < PATH_LENGTH:
            episodeStep += 1
            temp = int(episodeStep < TEMP_THRESHOLD)
            with torch.no_grad():
                pi = mcts.getActionProb(path, temp=temp)
            trainExamples.append([path, pi, None])
            vertex = np.random.choice(len(pi), p=pi)
            path = game.get_next_state(path, vertex)

        path_buffer.push(path)
        r = path_buffer.rank_path(path)
        for x in trainExamples:
            x[-1] = r
        train_buffer.push(trainExamples)

        #train phase
    
        for i in range(NUM_UPDATES):
            #sample batch from trainExamples
            batch = train_buffer.sample(BATCH_SIZE)
            paths, pis, vs = zip(*batch)
            embs = utils.get_states_emb(paths, graph_emb)

            states = torch.FloatTensor(np.array(embs).astype(np.float64))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            out_pi, out_v = nnet(embs)
            loss_pi = -torch.sum(target_pis*out_pi)/target_pis.size()[0]
            loss_v = torch.sum((targets_vs-out_v.view(-1))**2)/targets_vs.size()[0]
            total_loss = loss_pi + loss_v

            #record losses
            #pi_losses.update(l_pi.item(), boards.size(0))
            #v_losses.update(l_v.item(), boards.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()



