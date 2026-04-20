import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from astar_router import astar_router
from Arena_alpha_pd import Arena
from MCTS import MCTS
from tensorflow.nn import softmax

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.num_nets = args.num_nets
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.cleanerExamplesHistory = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, source, dest, init_board):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
                           
        """
        
        

        
        trainExamples = []
        board = self.game.getInitBoard(source, dest, init_board)
        self.curPlayer = 1
        episodeStep = 0
        r = self.game.getGameEnded(board, self.curPlayer)

        while r == 0:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            if self.game.getGameEnded(canonicalBoard, 1) == -1:
                pass
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            r = self.game.getGameEnded(board, self.curPlayer)

        if (episodeStep == 0) and r == 1:
            cleaner_reward = 0
        else:
            cleaner_reward = 1 - (episodeStep/(self.game.n * self.game.n)) - (25.0 * (0.5 - (0.5 *r)))
        return [(x[0], x[2], r) for x in trainExamples], cleaner_reward

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        star_board = astar_router(self.game.n)
        
        
        


        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                cleanerTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    nets_with_DRCs = set()
                    while len(nets_with_DRCs) == 0:
                            board_with_nets, numbers = star_board.get_board_with_connected_nets(self.num_nets)
                            nets_with_DRCs = star_board.get_nets_with_drc(board_with_nets)
                    
                    net_cleaner_rewards = []
                    for net in range(1, self.num_nets + 1): 
                        if net in nets_with_DRCs:
                            source, dest = numbers[net - 1]
                            init_board = star_board.convert_to_router_problem(board_with_nets, net)
                            TrainExamples, cleaner_reward = self.executeEpisode(source, dest, init_board)
                            iterationTrainExamples += TrainExamples
                            net_cleaner_rewards.append(cleaner_reward)
                        else:
                            net_cleaner_rewards.append(-25.0)
                    net_cleaner_rewards = softmax(net_cleaner_rewards).numpy()
                    cleanerTrainExamples += [(star_board.convert_to_cleaner_problem(board_with_nets), net_cleaner_rewards)]
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                self.cleanerExamplesHistory.append(cleanerTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
                self.cleanerExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            routeExamples = []
            for e in self.trainExamplesHistory:
                routeExamples.extend(e)
            shuffle(routeExamples)

            cleanerExamples = []
            for e in self.cleanerExamplesHistory:
                cleanerExamples.extend(e)
            shuffle(cleanerExamples)

            

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            

            self.nnet.train(routeExamples)
            self.nnet.train_cleaner(cleanerExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_router = os.path.join(folder, self.getCheckpointFile(iteration) + ".router.examples")
        with open(filename_router, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        filename_cleaner = os.path.join(folder, self.getCheckpointFile(iteration) + ".cleaner.examples")
        with open(filename_cleaner, "wb+") as f:
            Pickler(f).dump(self.cleanerExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        routerFile = modelFile + ".router.examples"
        if not os.path.isfile(routerFile):
            log.warning(f'File "{routerFile}" with routeExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with routeExamples found. Loading it...")
            with open(routerFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
        
        cleanerFile = modelFile + ".cleaner.examples"
        if not os.path.isfile(cleanerFile):
            log.warning(f'File "{cleanerFile}" with cleanExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with cleanExamples found. Loading it...")
            with open(cleanerFile, "rb") as f:
                self.cleanerExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
