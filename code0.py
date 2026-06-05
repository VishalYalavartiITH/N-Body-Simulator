import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class SimulationParams:
    """Configuration parameters for the N-Body simulation."""
    num_bodies: int = 10
    mass_range: Tuple[float, float] = (1.0, 5.0)  # kg
    position_range: Tuple[float, float] = (-45.0, 45.0)  # meters (within box)
    velocity_range: Tuple[float, float] = (-8.0, 8.0)  # m/s
    coefficient_of_restitution: float = 0.9  # Elasticity of collisions
    collision_distance: float = 2.0  # Detection radius
    G: float = 1.0  # Gravitational constant (reduced for 2D box)
    time_step: float = 0.05  # seconds
    total_time: float = 50.0  # seconds
    drag_coefficient: float = 0.01  # Small drag
    box_size: float = 100.0  # 2D bounded box size
    wall_restitution: float = 0.9  # Wall bounce elasticity


class Body:
    """Represents a single body in the N-Body system."""
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.acceleration = np.zeros(2)  # 2D only
        self.history = [self.position.copy()]

    def reset_acceleration(self):
        """Reset acceleration to zero before recalculation."""
        self.acceleration = np.zeros(2)

    def add_force(self, force: np.ndarray):
        """Add force (F=ma, so a=F/m)."""
        self.acceleration += force / self.mass

    def update_position(self, dt: float, drag: float = 0.0):
        """Update position using Verlet integration in 2D."""
        # Apply drag force
        if drag > 0:
            self.acceleration -= drag * self.velocity
        
        # Update velocity: v = v + a*dt
        self.velocity += self.acceleration * dt
        
        # Update position: x = x + v*dt + 0.5*a*dt^2
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        
        self.history.append(self.position.copy())


class NBodySimulator:
    """Direct N-Body gravitational simulator with collision detection in 2D bounded box."""
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.bodies = []
        self.time = 0.0
        self.collisions_count = 0
        self.wall_collisions_count = 0
        self._initialize_bodies()

    def _initialize_bodies(self):
        """Initialize bodies with random positions and velocities in 2D."""
        np.random.seed(42)  # For reproducibility
        half_box = self.params.box_size / 2
        
        for _ in range(self.params.num_bodies):
            mass = np.random.uniform(*self.params.mass_range)
            # Position in 2D within bounded box
            position = np.random.uniform(-half_box + 5, half_box - 5, size=2)
            velocity = np.random.uniform(*self.params.velocity_range, size=2)
            self.bodies.append(Body(mass, position, velocity))

    def calculate_gravitational_force(self, body1: Body, body2: Body) -> np.ndarray:
        """
        Calculate gravitational force on body1 due to body2 in 2D.
        F = G * m1 * m2 / r^2 * r_hat
        """
        r_vec = body2.position - body1.position
        r_magnitude = np.linalg.norm(r_vec)
        
        # Avoid singularity at r=0
        if r_magnitude < 1e-10:
            return np.zeros(2)
        
        r_hat = r_vec / r_magnitude
        magnitude = (self.params.G * body1.mass * body2.mass) / (r_magnitude**2)
        return magnitude * r_hat

    def detect_collision(self, body1: Body, body2: Body) -> bool:
        """Check if two bodies have collided."""
        distance = np.linalg.norm(body2.position - body1.position)
        return distance < self.params.collision_distance

    def resolve_collision(self, body1: Body, body2: Body):
        """
        Resolve collision using elastic collision formulas in 2D.
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
        
        # Impulse scalar (2D collision along contact normal)
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

    def handle_wall_collision(self, body: Body):
        """Handle collision with box walls."""
        half_box = self.params.box_size / 2
        e = self.params.wall_restitution
        
        # X boundaries
        if body.position[0] - 1.0 < -half_box:
            body.position[0] = -half_box + 1.0
            body.velocity[0] = abs(body.velocity[0]) * e
            self.wall_collisions_count += 1
        elif body.position[0] + 1.0 > half_box:
            body.position[0] = half_box - 1.0
            body.velocity[0] = -abs(body.velocity[0]) * e
            self.wall_collisions_count += 1
        
        # Y boundaries
        if body.position[1] - 1.0 < -half_box:
            body.position[1] = -half_box + 1.0
            body.velocity[1] = abs(body.velocity[1]) * e
            self.wall_collisions_count += 1
        elif body.position[1] + 1.0 > half_box:
            body.position[1] = half_box - 1.0
            body.velocity[1] = -abs(body.velocity[1]) * e
            self.wall_collisions_count += 1

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
        
        # Detect and resolve collisions between bodies
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                if self.detect_collision(body1, body2):
                    self.resolve_collision(body1, body2)
        
        # Update positions
        for body in self.bodies:
            body.update_position(self.params.time_step, self.params.drag_coefficient)
        
        # Handle wall collisions
        for body in self.bodies:
            self.handle_wall_collision(body)
        
        self.time += self.params.time_step

    def run_steps(self, num_steps: int):
        """Run specified number of simulation steps."""
        for _ in range(num_steps):
            self.step()


def animate_simulation(simulator: NBodySimulator, params: SimulationParams):
    """Create a real-time animation of the simulation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up the plot
    half_box = params.box_size / 2
    ax.set_xlim(-half_box - 5, half_box + 5)
    ax.set_ylim(-half_box - 5, half_box + 5)
    ax.set_aspect('equal')
    ax.set_title("2D N-Body Simulation - Real-time Collision")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, alpha=0.3)
    
    # Draw the box boundaries
    box = plt.Rectangle((-half_box, -half_box), params.box_size, params.box_size, 
                        fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Set up scatter plot for bodies
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulator.bodies)))
    scatter = ax.scatter([], [], s=[], c=[], alpha=0.7, edgecolors='black', linewidth=1)
    
    # Set up trajectory lines
    lines = [ax.plot([], [], color=colors[i], alpha=0.3, linewidth=0.5)[0] 
             for i in range(len(simulator.bodies))]
    
    # Text annotations
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    collision_text = ax.text(0.02, 0.93, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    steps_per_frame = 5  # Update every N simulation steps
    
    def animate(frame):
        # Run simulation steps
        simulator.run_steps(steps_per_frame)
        
        # Update positions and sizes
        positions = np.array([body.position for body in simulator.bodies])
        sizes = np.array([body.mass * 20 for body in simulator.bodies])
        
        scatter.set_offsets(positions)
        scatter.set_sizes(sizes)
        scatter.set_color(colors)
        
        # Update trajectory lines
        for i, body in enumerate(simulator.bodies):
            history = np.array(body.history)
            if len(history) > 1:
                lines[i].set_data(history[:, 0], history[:, 1])
        
        # Update text
        time_text.set_text(f'Time: {simulator.time:.2f}s')
        collision_text.set_text(f'Body Collisions: {simulator.collisions_count} | Wall Hits: {simulator.wall_collisions_count}')
        
        return scatter, *lines, time_text, collision_text
    
    # Calculate total frames needed
    total_steps = int(params.total_time / params.time_step)
    total_frames = total_steps // steps_per_frame
    
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=True, repeat=False)
    
    return fig, anim


def print_summary(params: SimulationParams, elapsed_time: float, collision_count: int, wall_collisions: int):
    """Print simulation summary and timing information."""
    print("\n" + "="*60)
    print("N-BODY SIMULATION SUMMARY")
    print("="*60)
    print(f"\nSimulation Parameters:")
    print(f"  • Number of Bodies: {params.num_bodies}")
    print(f"  • Box Size: {params.box_size}m × {params.box_size}m")
    print(f"  • Total Simulation Time: {params.total_time}s")
    print(f"  • Time Step: {params.time_step}s")
    print(f"  • Coefficient of Restitution: {params.coefficient_of_restitution}")
    print(f"  • Collision Detection Distance: {params.collision_distance}m")
    print(f"  • Drag Coefficient: {params.drag_coefficient}")
    print(f"\nResults:")
    print(f"  • Elapsed Computation Time: {elapsed_time:.4f}s")
    print(f"  • Body-to-Body Collisions: {collision_count}")
    print(f"  • Wall Collisions: {wall_collisions}")
    print(f"  • Simulation Speed: {params.total_time/elapsed_time:.2f}x real-time")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create custom parameters
    params = SimulationParams(
        num_bodies=10,
        mass_range=(1.0, 5.0),
        position_range=(-45.0, 45.0),
        velocity_range=(-8.0, 8.0),
        coefficient_of_restitution=0.9,
        collision_distance=2.0,
        time_step=0.05,
        total_time=50.0,
        drag_coefficient=0.01,
        box_size=100.0,
        wall_restitution=0.9
    )
    
    print("Starting N-Body Simulation with Real-Time Animation...")
    print(f"Configuration: {params.num_bodies} bodies in {params.box_size}m box")
    
    # Create simulator
    simulator = NBodySimulator(params)
    
    # Run animation
    start_time = time.time()
    fig, anim = animate_simulation(simulator, params)
    
    plt.tight_layout()
    plt.show()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print_summary(params, elapsed_time, simulator.collisions_count, simulator.wall_collisions_count)
    
    # Optionally save the final figure
    fig.savefig('nbody_simulation_final.png', dpi=150, bbox_inches='tight')
    print("Final frame saved to 'nbody_simulation_final.png'")
