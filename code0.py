import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class SimulationParams:
    """Configuration parameters for the N-Body simulation."""
    num_bodies: int = 5
    mass_range: Tuple[float, float] = (1.0, 10.0)  # kg
    position_range: Tuple[float, float] = (-100.0, 100.0)  # meters
    velocity_range: Tuple[float, float] = (-5.0, 5.0)  # m/s
    coefficient_of_restitution: float = 0.95  # Elasticity of collisions
    collision_distance: float = 5.0  # Detection radius
    G: float = 6.674e-11  # Gravitational constant
    time_step: float = 0.1  # seconds
    total_time: float = 100.0  # seconds
    drag_coefficient: float = 0.0  # No drag by default


class Body:
    """Represents a single body in the N-Body system."""
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.acceleration = np.zeros(3)
        self.history = [self.position.copy()]

    def reset_acceleration(self):
        """Reset acceleration to zero before recalculation."""
        self.acceleration = np.zeros(3)

    def add_force(self, force: np.ndarray):
        """Add force (F=ma, so a=F/m)."""
        self.acceleration += force / self.mass

    def update_position(self, dt: float, drag: float = 0.0):
        """Update position using Verlet integration."""
        # Apply drag force
        if drag > 0:
            self.acceleration -= drag * self.velocity
        
        # Update velocity: v = v + a*dt
        self.velocity += self.acceleration * dt
        
        # Update position: x = x + v*dt + 0.5*a*dt^2
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        
        self.history.append(self.position.copy())


class NBodySimulator:
    """Direct N-Body gravitational simulator with collision detection."""
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.bodies = []
        self.time = 0.0
        self.collisions_count = 0
        self._initialize_bodies()

    def _initialize_bodies(self):
        """Initialize bodies with random positions and velocities."""
        np.random.seed(42)  # For reproducibility
        for _ in range(self.params.num_bodies):
            mass = np.random.uniform(*self.params.mass_range)
            position = np.random.uniform(*self.params.position_range, size=3)
            velocity = np.random.uniform(*self.params.velocity_range, size=3)
            self.bodies.append(Body(mass, position, velocity))

    def calculate_gravitational_force(self, body1: Body, body2: Body) -> np.ndarray:
        """
        Calculate gravitational force on body1 due to body2.
        F = G * m1 * m2 / r^2 * r_hat
        """
        r_vec = body2.position - body1.position
        r_magnitude = np.linalg.norm(r_vec)
        
        # Avoid singularity at r=0
        if r_magnitude < 1e-10:
            return np.zeros(3)
        
        r_hat = r_vec / r_magnitude
        magnitude = (self.params.G * body1.mass * body2.mass) / (r_magnitude**2)
        return magnitude * r_hat

    def detect_collision(self, body1: Body, body2: Body) -> bool:
        """Check if two bodies have collided."""
        distance = np.linalg.norm(body2.position - body1.position)
        return distance < self.params.collision_distance

    def resolve_collision(self, body1: Body, body2: Body):
        """
        Resolve collision using elastic collision formulas.
        Assumes 1D collision along the line connecting centers.
        """
        self.collisions_count += 1
        
        # Vector from body1 to body2
        r_vec = body2.position - body1.position
        r_magnitude = np.linalg.norm(r_vec)
        
        if r_magnitude < 1e-10:
            return
        
        r_hat = r_vec / r_magnitude
        
        # Relative velocity
        v_rel = body2.velocity - body1.velocity
        v_rel_along_r = np.dot(v_rel, r_hat)
        
        # Only collide if approaching
        if v_rel_along_r >= 0:
            return
        
        # Coefficient of restitution
        e = self.params.coefficient_of_restitution
        
        # Impulse scalar (1D collision along contact normal)
        m1, m2 = body1.mass, body2.mass
        impulse_scalar = -(1 + e) * v_rel_along_r / (1/m1 + 1/m2)
        
        # Apply impulse
        impulse = impulse_scalar * r_hat
        body1.velocity -= impulse / m1
        body2.velocity += impulse / m2
        
        # Separate bodies to avoid overlap
        overlap = self.params.collision_distance - r_magnitude
        if overlap > 0:
            separation = (overlap / 2 + 0.1) * r_hat
            body1.position -= separation
            body2.position += separation

    def step(self):
        """Perform one simulation step."""
        # Reset accelerations
        for body in self.bodies:
            body.reset_acceleration()
        
        # Calculate gravitational forces
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                force = self.calculate_gravitational_force(body1, body2)
                body1.add_force(force)
                body2.add_force(-force)  # Newton's third law
        
        # Detect and resolve collisions
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                if self.detect_collision(body1, body2):
                    self.resolve_collision(body1, body2)
        
        # Update positions
        for body in self.bodies:
            body.update_position(self.params.time_step, self.params.drag_coefficient)
        
        self.time += self.params.time_step

    def run(self) -> Tuple[float, int]:
        """
        Run the complete simulation.
        Returns: (elapsed_time, collision_count)
        """
        start_time = time.time()
        steps = int(self.params.total_time / self.params.time_step)
        
        for step_num in range(steps):
            self.step()
            if (step_num + 1) % max(1, steps // 10) == 0:
                print(f"  Progress: {((step_num + 1) / steps * 100):.1f}%")
        
        elapsed_time = time.time() - start_time
        return elapsed_time, self.collisions_count


def visualize_simulation(simulator: NBodySimulator):
    """Create a 2D visualization of the simulation trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectories
    ax1.set_title("N-Body Trajectories (XY Plane)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True, alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulator.bodies)))
    
    for i, body in enumerate(simulator.bodies):
        history = np.array(body.history)
        ax1.plot(history[:, 0], history[:, 1], 'o-', color=colors[i], 
                label=f"Body {i+1} (m={body.mass:.1f}kg)", markersize=2, linewidth=0.8)
        # Mark final position
        ax1.plot(history[-1, 0], history[-1, 1], 'o', color=colors[i], markersize=8)
    
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    
    # Plot 2: 3D view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("N-Body Trajectories (3D)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_zlabel("Z Position (m)")
    
    for i, body in enumerate(simulator.bodies):
        history = np.array(body.history)
        ax2.plot(history[:, 0], history[:, 1], history[:, 2], 
                color=colors[i], label=f"Body {i+1}", linewidth=0.8)
    
    plt.tight_layout()
    return fig


def print_summary(params: SimulationParams, elapsed_time: float, collision_count: int):
    """Print simulation summary and timing information."""
    print("\n" + "="*60)
    print("N-BODY SIMULATION SUMMARY")
    print("="*60)
    print(f"\nSimulation Parameters:")
    print(f"  • Number of Bodies: {params.num_bodies}")
    print(f"  • Total Simulation Time: {params.total_time}s")
    print(f"  • Time Step: {params.time_step}s")
    print(f"  • Coefficient of Restitution: {params.coefficient_of_restitution}")
    print(f"  • Collision Detection Distance: {params.collision_distance}m")
    print(f"  • Drag Coefficient: {params.drag_coefficient}")
    print(f"\nResults:")
    print(f"  • Elapsed Computation Time: {elapsed_time:.4f}s")
    print(f"  • Total Collisions Detected: {collision_count}")
    print(f"  • Simulation Speed: {params.total_time/elapsed_time:.2f}x real-time")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create custom parameters
    params = SimulationParams(
        num_bodies=5,
        mass_range=(1.0, 10.0),
        position_range=(-100.0, 100.0),
        velocity_range=(-5.0, 5.0),
        coefficient_of_restitution=0.95,
        collision_distance=5.0,
        time_step=0.1,
        total_time=100.0,
        drag_coefficient=0.0  # No drag
    )
    
    print("Starting N-Body Simulation...")
    print(f"Configuration: {params.num_bodies} bodies, {params.total_time}s simulation time")
    
    # Create and run simulator
    simulator = NBodySimulator(params)
    elapsed_time, collision_count = simulator.run()
    
    # Print timing and results
    print_summary(params, elapsed_time, collision_count)
    
    # Visualize results
    fig = visualize_simulation(simulator)
    plt.savefig('nbody_simulation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'nbody_simulation.png'")
    plt.show()
