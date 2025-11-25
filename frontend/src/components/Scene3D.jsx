import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, MeshDistortMaterial, Sphere } from '@react-three/drei'
import * as THREE from 'three'

function FloatingShape({ position, scale, speed, color }) {
  const meshRef = useRef()

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    meshRef.current.rotation.x = time * speed * 0.3
    meshRef.current.rotation.y = time * speed * 0.5
    meshRef.current.position.y = position[1] + Math.sin(time * speed) * 0.5
  })

  return (
    <mesh ref={meshRef} position={position} scale={scale}>
      <icosahedronGeometry args={[1, 0]} />
      <meshStandardMaterial
        color={color}
        wireframe
        transparent
        opacity={0.15}
      />
    </mesh>
  )
}

function AnimatedSphere({ position }) {
  const meshRef = useRef()

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    meshRef.current.rotation.z = time * 0.2
  })

  return (
    <Sphere ref={meshRef} args={[1, 64, 64]} position={position} scale={1.5}>
      <MeshDistortMaterial
        color="#0ea5e9"
        attach="material"
        distort={0.3}
        speed={1.5}
        roughness={0.4}
        wireframe
        transparent
        opacity={0.1}
      />
    </Sphere>
  )
}

function ParticleField() {
  const pointsRef = useRef()

  const particles = useMemo(() => {
    const count = 100
    const positions = new Float32Array(count * 3)

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20
    }

    return positions
  }, [])

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    pointsRef.current.rotation.y = time * 0.05
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.length / 3}
          array={particles}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#0ea5e9"
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  )
}

export default function Scene3D() {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas
        camera={{ position: [0, 0, 10], fov: 50 }}
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} color="#0ea5e9" />

        {/* Floating geometric shapes */}
        <FloatingShape position={[-4, 2, -3]} scale={0.6} speed={0.5} color="#0ea5e9" />
        <FloatingShape position={[4, -2, -4]} scale={0.4} speed={0.7} color="#38bdf8" />
        <FloatingShape position={[-3, -3, -5]} scale={0.5} speed={0.6} color="#7dd3fc" />
        <FloatingShape position={[3, 3, -3]} scale={0.3} speed={0.8} color="#0284c7" />

        {/* Animated sphere */}
        <AnimatedSphere position={[0, 0, -5]} />

        {/* Particle field */}
        <ParticleField />

        {/* Optional: Enable orbit controls for debugging */}
        {/* <OrbitControls enableZoom={false} enablePan={false} /> */}
      </Canvas>
    </div>
  )
}
