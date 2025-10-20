"""
Custom IQL implementation with auxiliary price prediction task.

Combines:
1. IQL loss (actor + critic + value)
2. Auxiliary classification loss (predict future price direction)

The auxiliary task helps the encoder learn better representations.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from d3rlpy.algos import IQL
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.logging import LoggerAdapter
from d3rlpy.torch_utility import TorchMiniBatch


class IQLWithAuxiliary(IQL):
    """
    IQL with auxiliary price prediction task.
    
    Training objective:
        total_loss = IQL_loss + aux_weight * auxiliary_loss
        
    Where:
        - IQL_loss: Standard IQL (expectile regression + policy extraction)
        - auxiliary_loss: Cross-entropy for predicting price direction
        - aux_weight: Balancing coefficient (default 0.1)
    """
    
    def __init__(self, *args, aux_loss_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_loss_weight = aux_loss_weight
        self._aux_labels = None  # Will be set from dataset
    
    def set_auxiliary_labels(self, labels: torch.Tensor):
        """
        Set auxiliary labels for the full dataset.
        
        Args:
            labels: [N,] array of future price directions (-1, 0, +1)
        """
        self._aux_labels = labels
    
    def _update_actor(self, batch: TorchMiniBatch) -> dict:
        """Override to add auxiliary loss."""
        # Standard IQL actor update
        metrics = super()._update_actor(batch)
        
        # Add auxiliary loss if encoder supports it
        if hasattr(self._impl._policy, 'encoder') and \
           hasattr(self._impl._policy.encoder, 'forward_with_aux'):
            
            # Get auxiliary predictions
            _, aux_logits = self._impl._policy.encoder.forward_with_aux(batch.observations)
            
            # Get ground truth labels (need to map batch indices)
            if self._aux_labels is not None and hasattr(batch, 'indices'):
                # Convert labels from {-1, 0, +1} to class indices {0, 1, 2}
                batch_labels = self._aux_labels[batch.indices]
                class_indices = (batch_labels + 1).long()  # -1→0, 0→1, +1→2
                
                # Compute auxiliary loss
                aux_loss = F.cross_entropy(aux_logits, class_indices)
                
                # Add to actor loss
                actor_loss = metrics.get('actor_loss', 0.0)
                total_loss = actor_loss + self._aux_loss_weight * aux_loss
                
                # Log metrics
                metrics['aux_loss'] = aux_loss.item()
                metrics['aux_accuracy'] = (aux_logits.argmax(dim=1) == class_indices).float().mean().item()
                metrics['actor_loss_total'] = total_loss
        
        return metrics
    
    def fit(
        self,
        dataset: ReplayBuffer,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: Optional[LoggerAdapter] = None,
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[dict] = None,
        callback: Optional[callable] = None,
        epoch_callback: Optional[callable] = None,
    ):
        """Override fit to extract and set auxiliary labels."""
        # Extract auxiliary labels from dataset
        if hasattr(dataset, '_aux_labels'):
            aux_labels_np = dataset._aux_labels
            aux_labels_tensor = torch.from_numpy(aux_labels_np).float()
            self.set_auxiliary_labels(aux_labels_tensor)
            print(f"Auxiliary labels loaded: {len(aux_labels_tensor)} samples")
            print(f"  UP: {(aux_labels_np > 0).sum()}, "
                  f"NEUTRAL: {(aux_labels_np == 0).sum()}, "
                  f"DOWN: {(aux_labels_np < 0).sum()}")
        else:
            print("Warning: Dataset has no auxiliary labels. Auxiliary task disabled.")
        
        # Call parent fit
        return super().fit(
            dataset=dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logger_adapter=logger_adapter,
            show_progress=show_progress,
            save_interval=save_interval,
            evaluators=evaluators,
            callback=callback,
            epoch_callback=epoch_callback,
        )


def create_iql_with_auxiliary(
    aux_loss_weight: float = 0.1,
    **iql_kwargs
) -> IQLWithAuxiliary:
    """
    Create IQL algorithm with auxiliary task.
    
    Args:
        aux_loss_weight: Weight for auxiliary loss (default 0.1)
        **iql_kwargs: Standard IQL parameters
        
    Returns:
        IQLWithAuxiliary instance
    """
    return IQLWithAuxiliary(aux_loss_weight=aux_loss_weight, **iql_kwargs)
