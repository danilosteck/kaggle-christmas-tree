import numpy as np
import random
from typing import List, Tuple, Dict, Callable
from christmas_tree.utils import initialize_compact_population
import logging

logs_path = os.path.join('.','christmas_tree','logs')
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

# Configuração para saída em um arquivo
logging.basicConfig(filename=os.path.join(logs_path,'results_swarm.log'), level=logging.INFO, format='%(asctime)s; %(message)s')
# Mensagem de log

class ParticleSwarmOptimization:
    def __init__(self, 
                 num_shapes: int,
                 bounds_ranges: List[Tuple[float, float]],
                 population_size: int = 50,
                 iterations: int = 200,
                 w: float = 0.7,      # Inertia
                 c1: float = 1.5,     # Cognitive
                 c2: float = 1.5,     # Social
                 evaluate_func: Callable = None,
                 print_generations: int=20 # Print results after each X generations - default 20 generations
                 ):
        self.num_shapes = num_shapes
        self.bounds_ranges = bounds_ranges
        self.swarm_size = population_size
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.evaluate = evaluate_func
        self.print_generations = print_generations
        
        # Initialize swarm
        self.particles = self._initialize_swarm()
        self.velocities = self._initialize_velocities()
        self.pbest_positions = self.particles.copy()
        self.pbest_scores = [float('inf')] * self.swarm_size
        self.gbest_position = None
        self.gbest_score = float('inf')

    def _initialize_swarm(self):
        compact_population = self._compact_population(max(3, np.log(self.num_shapes)))
    # Create swarm: each particle starts from base + small random noise
        swarm = []
        for _ in range(self.swarm_size):  # swarm_size
            particle = []
            for pos in compact_population:
                # Add small noise for diversity (±1 unit, ±5°)
                noise_x = random.gauss(0, 0.5)
                noise_y = random.gauss(0, 0.5)
                noise_deg = random.gauss(0, 2.0)
                
                new_x = pos[0] + noise_x
                new_y = pos[1] + noise_y
                new_deg = (pos[2] + noise_deg) % 360
                
                particle.append((new_x, new_y, new_deg))
            
            swarm.append(particle)
        
        return swarm
        
    # def _initialize_swarm(self) -> List[List[Tuple[float, float, float]]]:
    #     """Initialize particle positions."""
    #     swarm = []
    #     for _ in range(self.swarm_size):
    #         particle = []
    #         for i in range(self.num_shapes):
    #             x = random.uniform(*self.bounds_ranges[0])
    #             y = random.uniform(*self.bounds_ranges[1])
    #             deg = random.uniform(*self.bounds_ranges[2])
    #             particle.append((x, y, deg))
    #         swarm.append(particle)
    #     return swarm
    
    def _initialize_velocities(self) -> List[List[float]]:
        """Initialize velocities (3D per shape: vx,vy,vdeg)."""
        velocities = []
        for _ in range(self.swarm_size):
            vel = []
            for i in range(self.num_shapes):
                vx = random.uniform(-1.0, 1.0)
                vy = random.uniform(-1.0, 1.0)
                vdeg = random.uniform(-5.0, 5.0)
                vel.extend([vx, vy, vdeg])
            velocities.append(vel)
        return velocities

    def _compact_population(self, scale:float) -> dict:
        # Cria uma população com empacotamento compacto determinístico, com espaçamento pré-definido e usando um fator de escala para aumentar o espaço de soluções possíveis.
        from christmas_tree.utils import initialize_compact_population
        compact_population = initialize_compact_population(self.num_shapes,.705,.8,scale)
        return compact_population
    
    def _evaluate_particle(self, particle: List[Tuple[float, float, float]]) -> Dict:
        from christmas_tree.utils import tuple_to_kaggle_output
        result = self.evaluate(tuple_to_kaggle_output(particle))
        result['is_valid'] = not result['has_violations']
        return result
    
    def _update_velocity(self, particle_idx: int):
        """PSO velocity update: v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)"""
        particle = self.particles[particle_idx]
        velocity = self.velocities[particle_idx]
        
        # Flatten particle for velocity calculation
        pos_flat = []
        for x, y, deg in particle:
            pos_flat.extend([x, y, deg])
            
        pbest_flat = []
        for x, y, deg in self.pbest_positions[particle_idx]:
            pbest_flat.extend([x, y, deg])
            
        gbest_flat = []
        for x, y, deg in self.gbest_position:
            gbest_flat.extend([x, y, deg])
        
        r1 = np.random.random(3 * self.num_shapes)
        r2 = np.random.random(3 * self.num_shapes)
        
        # Velocity update
        new_velocity = (
            self.w * np.array(velocity) +
            self.c1 * r1 * (np.array(pbest_flat) - np.array(pos_flat)) +
            self.c2 * r2 * (np.array(gbest_flat) - np.array(pos_flat))
        )
        
        # Clamp velocities
        max_vel = [2.0, 2.0, 10.0] * self.num_shapes
        new_velocity = np.clip(new_velocity, np.negative(max_vel), max_vel)
        
        self.velocities[particle_idx] = new_velocity.tolist()
    
    def _update_position(self, particle_idx: int):
        """Update position using velocity, clamp to bounds."""
        particle = self.particles[particle_idx]
        velocity = self.velocities[particle_idx]
        
        new_particle = []
        x_range, y_range, deg_range = self.bounds_ranges
        
        vel_idx = 0
        for i in range(self.num_shapes):
            x, y, deg = particle[i]
            vx, vy, vdeg = velocity[vel_idx:vel_idx+3]
            
            new_x = np.clip(x + vx, *x_range)
            new_y = np.clip(y + vy, *y_range)
            new_deg = np.clip((deg + vdeg) % 360, *deg_range)
            
            new_particle.append((new_x, new_y, new_deg))
            vel_idx += 3
        
        self.particles[particle_idx] = new_particle
    
    def run(self) -> Dict:
        """Run PSO optimization."""
        # Initial evaluation
        import uuid
        run_id = uuid.uuid4()

        has_valid_result = False
        for i in range(self.swarm_size):
            result = self._evaluate_particle(self.particles[i])
            score = result['score']
            
            # Update pbest
            if result['is_valid'] and score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest_positions[i] = self.particles[i][:]
                has_valid_result = True
            
            # Update gbest
            if result['is_valid'] and score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = self.particles[i][:]
                has_valid_result = True

            # TODO: Refazer para que, quando não encontre uma solução inicial, use uma população compacta
            # pop = self._compact_population(3.5)
            # cp_result = self._evaluate_particle(pop)
            # print(cp_result)
            # if not has_valid_result:
            #     self.pbest_scores[i] = cp_result['score']
            #     self.pbest_positions[i] = pop
            #     self.gbest_score = cp_result['score']
            #     self.gbest_positions = pop
        
        print(f"Initial gbest: {self.gbest_score:.2f}")
        
        # Main iterations
        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                # Update velocity and position
                self._update_velocity(i)
                self._update_position(i)
                
                # Evaluate new position
                result = self._evaluate_particle(self.particles[i])
                has_valid_reuslt = False
                
                # Update pbest
                if result['is_valid'] and result['score'] < self.pbest_scores[i]:
                    self.pbest_scores[i] = result['score']
                    self.pbest_positions[i] = self.particles[i][:]
                    has_valid_result = True
                
                # Update gbest
                if result['is_valid'] and result['score'] < self.gbest_score:
                    self.gbest_score = result['score']
                    self.gbest_position = self.particles[i][:]
                    has_valid_result = True

                #             # Update range:
                # if result['is_valid']:
                #     self.bounds_ranges = [
                #         [(pd.DataFrame(self.gbest_position)[0].min()), (pd.DataFrame(self.gbest_position)[0].max())],
                #         [(pd.DataFrame(self.gbest_position)[1].min()), (pd.DataFrame(self.gbest_position)[1].max())],
                #         [-180,180]
                #     ]
            
            if iteration % self.print_generations == 0:
                print(f"Iter {iteration}: gbest = {self.gbest_score:.2f}; has valid result: {has_valid_result}")
                logging.info(f'{iteration};{run_id};Swarm Optimization;{self.num_shapes};{self.swarm_size};{self.iterations};{self.bounds_ranges};{self.w};{self.c1};{self.c2};{self.gbest_score};{has_valid_result}')


        # self.num_shapes = num_shapes
        # self.bounds_ranges = bounds_ranges
        # self.swarm_size = population_size
        # self.iterations = iterations
        # self.w, self.c1, self.c2 = w, c1, c2
        
        final_result = self._evaluate_particle(self.gbest_position)
        return {
            'num_shapes': self.num_shapes,
            'bounds_ranges': self.bounds_ranges,
            'params':{
                'iterations': self.iterations,
                'swarm_size': self.swarm_size,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2
            },
            'best_individual': self.gbest_position,
            'best_score': self.gbest_score,
            **final_result
        }

# Usage: Replace GeneticAlgorithm with ParticleSwarmOptimization
# pso = ParticleSwarmOptimization(num_shapes=3, bounds_ranges=var_ranges, evaluate_func=your_evaluate)
# result = pso.run()

if __name__ == '__main__':
    from christmas_tree.kaggle_source import *
    from christmas_tree.plot_trees import plot_trees
    from christmas_tree.utils import tuple_to_kaggle_output


    pso = ParticleSwarmOptimization(
        num_shapes=33, 
        bounds_ranges=[(-100,100), (-100,100), (-180,180)], 
        evaluate_func=score_mod,
        population_size=100,
        iterations=100,
        w=0.5,  
        c1=1.5,  
        c2=1.5,  
        print_generations=1 # Print results after each X generations - default 20 generations
    )

    result = pso.run()

    import itertools

    pop_size_range = [50]
    iterations = [301]
    w = [0.5, 0.75, 1]
    c1 = [1.5, 2.5]
    c2 = [1.5, 2.5]

    combinations = list(itertools.product(pop_size_range, iterations, w, c1, c2))

    i = 0
    for c in combinations:
        i+=1
        print(f"Combination {i} out of {len(combinations)}; {round(i/len(combinations),4)*100}%")
        num_shapes = 10
        for k in range(0,3):
            pso = ParticleSwarmOptimization(
                num_shapes=num_shapes, 
                bounds_ranges=[(-100,100), (-100,100), (-180,180)], 
                evaluate_func=score_mod,
                population_size=c[0]*num_shapes,
                iterations=c[1],
                w=c[2],  
                c1=c[3],  
                c2=c[4],  
                print_generations=10 # Print results after each X generations - default 20 generations
            )

            result = pso.run()
            
    # plot_trees(tuple_to_kaggle_output(result['best_individual']))

    # plot_trees(tuple_to_kaggle_output([(0,0,0),(0.75,0,0),(0,1,0),(0.75,1,0)]))

    # dx = 0.7
    # dy = 0.8
    # ddeg = -180
    # pos = [(0,0,0),(dx,0,0),(dx/2,dy,ddeg),(dx+dx/2,dy,ddeg)]
    # plot_trees(tuple_to_kaggle_output(pos))

    # dx = 1.4
    # dy = 0.0
    # ddeg = 0
    # pos = [(0,0,0),(dx/2,dy,ddeg)]
    # score_mod(tuple_to_kaggle_output(pos))
    # plot_trees(tuple_to_kaggle_output(pos))


    # dx = 1.4
    # dy = 0.0
    # ddeg = 0
    # pos = [(0,0,0),(dx/2,dy,ddeg),(dx,dy,ddeg)]
    # score_mod(tuple_to_kaggle_output(pos))
    # plot_trees(tuple_to_kaggle_output(pos))

    # dx = 1.4
    # dy = 0.0
    # ddeg = 0
    # pos = [(0,0,0),(dx/2,dy,ddeg),(dx,dy,ddeg),(dx*3/2,dy,ddeg)]
    # score_mod(tuple_to_kaggle_output(pos))
    # plot_trees(tuple_to_kaggle_output(pos))

