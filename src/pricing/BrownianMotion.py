import numpy as np
import random 

class BrownianMotion:
    """
    Gère la génération de mouvements Browniens vectoriels.
    """
    def __init__(self, seed: int = 1234):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_dw(self, T: float, nb_steps: int, nb_paths: int, antithetic: bool = True) -> tuple[np.ndarray, float]:
        """
        Génère une matrice d'incréments Browniens (dW).
        
        Retourne:
            dW: numpy array de taille (nb_paths, nb_steps)
            dt: pas de temps calculé
        """
        dt = T / nb_steps
        sqrt_dt = np.sqrt(dt)
        actual_draws = nb_paths // 2 if antithetic else nb_paths

        # 1. Tirage Normal Standard Z ~ N(0, 1)
        Z = self.rng.standard_normal((actual_draws, nb_steps))

        # 2. Gestion Antithétique (W et -W)
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)
            # Si nb_paths était impair, on s'assure d'avoir le bon compte (optionnel, souvent nb_paths est pair)
            if Z.shape[0] < nb_paths:
                # On comble le manque (cas rare si nb_paths est pair)
                extra = self.rng.standard_normal((nb_paths - Z.shape[0], nb_steps))
                Z = np.concatenate([Z, extra], axis=0)

        # 3. Conversion en incréments Browniens dW = Z * sqrt(dt)
        dW = Z * sqrt_dt
        
        return dW, dt

    def next_gaussian(self) -> float:
        """
        Version Scalaire : Retourne UN seul tirage N(0,1).
        Correspond aux slides: 'Get one random draw'
        """
        return self.rng.standard_normal()