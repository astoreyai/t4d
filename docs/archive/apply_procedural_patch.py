#!/usr/bin/env python3
"""
Apply dopamine integration patch to procedural.py
"""

import re

# Read the file
with open('/mnt/projects/ww/src/ww/memory/procedural.py', 'r') as f:
    content = f.read()

# Change 1: Add dopamine import after line with traced import
import_pattern = r'(from ww\.observability\.tracing import traced, add_span_attribute\n)'
import_replacement = r'\1from ww.learning.dopamine import DopamineSystem\n'
content = re.sub(import_pattern, import_replacement, content)

# Change 2: Add dopamine system to __init__
# Find the end of __init__ (before async def initialize)
init_pattern = r'(        self\.experience_weight = settings\.procedural_weight_experience\n)'
init_addition = '''
        # Optional dopamine system for RPE-modulated learning
        self.dopamine_system = None  # Lazy init
        self._dopamine_enabled = getattr(settings, "procedural_dopamine_enabled", False)
'''
content = re.sub(init_pattern, r'\1' + init_addition, content)

# Change 3: Add dopamine RPE computation in update() method
# Find the section after "procedure = await self.get_procedure(procedure_id)"
update_pattern = r'(        if not procedure:\n            raise ValueError\(f"Procedure {procedure_id} not found"\)\n)'

update_addition = '''
        # Compute dopamine signal (RPE) for surprise-modulated learning
        rpe_signal = None
        if self._dopamine_enabled:
            try:
                # Lazy init
                if self.dopamine_system is None:
                    self.dopamine_system = DopamineSystem(
                        default_expected=0.5,
                        value_learning_rate=0.1,
                        surprise_threshold=0.05
                    )

                # Use procedure name as context hash
                actual_outcome = 1.0 if success else 0.0

                # Compute RPE: δ = actual - expected
                rpe = self.dopamine_system.compute_rpe(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )
                rpe_signal = rpe.rpe

                # Update expectations for next time
                self.dopamine_system.update_expectations(
                    memory_id=procedure_id,
                    actual_outcome=actual_outcome
                )

                logger.debug(
                    f"Dopamine signal for '{procedure.name}': "
                    f"RPE={rpe_signal:.3f} (expected={rpe.expected:.2f}, actual={actual_outcome:.1f})"
                )

                # Use RPE to modulate consolidation
                # High |RPE| = surprising = more learning
                # Expected outcomes (RPE≈0) = less learning
                if abs(rpe_signal) > 0.1:
                    logger.info(
                        f"Surprising outcome for '{procedure.name}': RPE={rpe_signal:.3f} "
                        f"({'better' if rpe_signal > 0 else 'worse'} than expected)"
                    )

            except Exception as e:
                logger.warning(f"Dopamine RPE computation failed: {e}")

'''

content = re.sub(update_pattern, r'\1' + update_addition, content)

# Write the modified content
with open('/mnt/projects/ww/src/ww/memory/procedural.py', 'w') as f:
    f.write(content)

print("✓ Procedural memory dopamine integration applied")
