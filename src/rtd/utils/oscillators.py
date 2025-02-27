import time
import math
import numpy as np
from typing import Dict, Tuple, Literal, Union, List


class Oscillator:
    """
    A class that provides various oscillation patterns.
    
    This class can generate continuous sine waves or trigger pulses
    based on the specified parameters.
    """
    
    def __init__(self):
        # Store oscillator states: (start_time, period, min_value, max_value, mode, [last_cycle])
        self.oscillators: Dict[str, Union[Tuple[float, float, float, float, str], Tuple[float, float, float, float, str, int]]] = {}
    
    def get(self, 
            name: str, 
            period: float, 
            min_value: float, 
            max_value: float, 
            mode: Literal['continuous', 'trigger'] = 'continuous') -> float:
        """
        Get the current value of an oscillator.
        
        Args:
            name: Unique identifier for this oscillator
            period: Time in seconds for a complete cycle
            min_value: Minimum value in the oscillation range
            max_value: Maximum value in the oscillation range
            mode: 'continuous' for sine wave, 'trigger' for periodic pulse
            
        Returns:
            float: Current value of the oscillator
        """
        current_time = time.time()
        
        # Register oscillator if it doesn't exist
        if name not in self.oscillators:
            if mode == 'continuous':
                self.oscillators[name] = (current_time, period, min_value, max_value, mode)
            else:  # 'trigger' mode
                self.oscillators[name] = (current_time, period, min_value, max_value, mode, -1)
        
        # Get oscillator parameters
        oscillator_data = self.oscillators[name]
        start_time = oscillator_data[0]
        stored_period = oscillator_data[1]
        stored_min = oscillator_data[2]
        stored_max = oscillator_data[3]
        stored_mode = oscillator_data[4]
        
        # Update parameters if they've changed
        if (stored_period != period or stored_min != min_value or 
            stored_max != max_value or stored_mode != mode):
            if mode == 'continuous':
                self.oscillators[name] = (start_time, period, min_value, max_value, mode)
            else:  # 'trigger' mode
                last_cycle = oscillator_data[5] if len(oscillator_data) > 5 else -1
                self.oscillators[name] = (start_time, period, min_value, max_value, mode, last_cycle)
        
        # Calculate phase (0 to 1)
        elapsed_time = current_time - start_time
        phase = (elapsed_time % period) / period

        print(f'phase {phase}')
        
        if mode == 'continuous':
            # Sine wave oscillation
            amplitude = (max_value - min_value) / 2
            offset = min_value + amplitude
            return offset + amplitude * math.sin(2 * math.pi * phase)
        
        elif mode == 'trigger':
            # Return non-zero only once at the start of each cycle
            # Store the cycle count to detect new cycles reliably
            elapsed_time = current_time - start_time
            current_cycle = int(elapsed_time / period)
            
            # Get the stored cycle
            last_cycle = oscillator_data[5]
            
            # Update the tuple with the current cycle
            if current_cycle > last_cycle:
                self.oscillators[name] = (start_time, period, min_value, max_value, mode, current_cycle)
                return 1.0
                
            return 0.0
        
        # Default fallback
        return min_value
