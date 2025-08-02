from datetime import datetime, timedelta, date

class SimulationTime:
    def __init__(self, start_date_str: str, epoch_duration_days: int):
        try:
            # This is the start datetime of the very first epoch (epoch 0)
            self.simulation_start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid start_date format: {start_date_str}. Expected YYYY-MM-DD.")
        
        if not isinstance(epoch_duration_days, int) or epoch_duration_days <= 0:
            raise ValueError("epoch_duration_days must be a positive integer.")
        
        self.epoch_duration = timedelta(days=epoch_duration_days)
        # current_time will represent the start of the current epoch
        self.current_time = self.simulation_start_datetime
        self.current_epoch = 0 

    def advance_epoch(self):
        self.current_time += self.epoch_duration
        self.current_epoch += 1

    def get_current_time_str(self, fmt="%Y-%m-%d %H:%M:%S") -> str:
        return self.current_time.strftime(fmt)

    def get_current_epoch(self) -> int:
        return self.current_epoch

    def get_epoch_from_date(self, target_datetime: datetime) -> int:
        """
        Calculates the epoch number for a given target_datetime based on the simulation start and epoch duration.
        """
        if not isinstance(target_datetime, datetime):
            # If it's a date object, convert to datetime at midnight for consistent comparison
            if isinstance(target_datetime, date):
                target_datetime = datetime.combine(target_datetime, datetime.min.time())
            else:
                raise TypeError(f"target_datetime must be a datetime object, got {type(target_datetime)}")

        if target_datetime < self.simulation_start_datetime:
            # Handle dates before the simulation starts; could be epoch 0 or an error/special value
            # For simplicity, let's consider it part of the "zeroth" period or raise an error if unexpected.
            # Returning 0 might be problematic if epoch 0 has a defined start.
            # Let's assume posts are not expected before simulation_start_datetime for epoch calculation.
            # If they are, the definition of epoch -1 or similar might be needed.
            # For now, clamping to epoch 0 or first valid epoch.
             return 0 # Or consider raising ValueError("Date is before simulation start")

        if self.epoch_duration.total_seconds() <= 0:
            # This case should be prevented by __init__ checks but good for robustness
            raise ValueError("Epoch duration must be positive.")

        time_difference = target_datetime - self.simulation_start_datetime
        
        # Calculate how many full epoch_durations fit into the time_difference
        # Integer division of total_seconds ensures we get the floor value,
        # which corresponds to the epoch number (0-indexed).
        epoch_number = int(time_difference.total_seconds() // self.epoch_duration.total_seconds())
        
        return epoch_number

    def __repr__(self) -> str:
        return f"SimulationTime(current_epoch={self.current_epoch}, current_time='{self.get_current_time_str()}', simulation_start='{self.simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}')"





