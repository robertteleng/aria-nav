# Problem-Solving con 4 Frameworks

## Descripción General

Esta guía proporciona un sistema integrado de frameworks mentales para abordar cualquier problema técnico, proyecto de desarrollo, o desafío de aprendizaje de manera sistemática y eficiente.

## Los 4 Frameworks Core

### 1. WOOP - Session Planning
**Propósito:** Planificación y setup de sesiones de trabajo

- **W** - **Wish**: ¿Qué quiero lograr?
- **O** - **Outcome**: ¿Cómo sabré que lo conseguí?
- **O** - **Obstacles**: ¿Qué puede salir mal?
- **P** - **Plan**: ¿Cuál es mi approach específico?

### 2. OODA Loop - Active Problem Solving
**Propósito:** Resolución de problemas específicos en tiempo real

- **O** - **Observe**: ¿Qué está pasando exactamente?
- **O** - **Orient**: ¿Cómo se relaciona con lo que sé?
- **D** - **Decide**: ¿Cuál es la mejor acción?
- **A** - **Act**: Ejecutar y medir resultado

### 3. SMART - Continuous Learning Capture
**Propósito:** Capturar aprendizaje y conocimiento mientras trabajas

- **S** - **State**: ¿Dónde estoy ahora?
- **M** - **Make observations**: ¿Qué veo/noto?
- **A** - **Ask questions**: ¿Por qué? ¿Cómo?
- **R** - **Relate**: ¿Con qué conceptos conecta?
- **T** - **Take action**: ¿Próximos pasos?

### 4. AAR - Session Review
**Propósito:** Revisión y mejora continua al final de sesiones

- **A** - **Actions**: ¿Qué pasó realmente?
- **A** - **Assessment**: ¿Qué fue bien/mal?
- **R** - **Recommendations**: ¿Qué cambiar la próxima vez?

## Cuándo Usar Cada Framework

### WOOP - Al Inicio de Cualquier Sesión
```
✓ Empezar nuevo proyecto
✓ Comenzar nueva funcionalidad
✓ Abordar problema complejo
✓ Sesión de aprendizaje enfocada
```

### OODA Loop - Durante Resolución Activa
```
✓ Error bloqueante aparece
✓ Decisión técnica necesaria
✓ Optimización específica requerida
✓ Debugging de problema concreto
✓ Arquitectura decision needed
```

### SMART - Observación Continua
```
✓ Descubrir concepto nuevo
✓ Ver conexión interesante
✓ Notar pattern emergente
✓ Insight sobre tecnología
✓ Learning moment significativo
```

### AAR - Final de Sesión/Milestone
```
✓ Completar funcionalidad
✓ Resolver problema mayor
✓ Final del día de trabajo
✓ Milestone del proyecto alcanzado
```

## Framework Integration Workflow

### Flujo Completo de una Sesión de Desarrollo

```
START SESSION:
│
├─ WOOP (Planning)
│  ├─ Define wish/outcome for session
│  ├─ Identify potential obstacles  
│  └─ Create specific action plan
│
├─ DEVELOPMENT LOOP:
│  │
│  ├─ OODA CYCLE (per problem encountered):
│  │  ├─ Observe what's happening
│  │  ├─ Orient to existing knowledge
│  │  ├─ Decide on approach
│  │  ├─ Act and measure result
│  │  └─ Repeat until resolved
│  │
│  ├─ SMART CAPTURE (continuous):
│  │  ├─ State current situation
│  │  ├─ Make observations about what works/doesn't
│  │  ├─ Ask deeper questions about why/how
│  │  ├─ Relate to broader concepts and patterns
│  │  └─ Take note of next steps/implications
│  │
│  └─ Return to OODA for next problem
│
└─ END SESSION:
   │
   └─ AAR (Review)
      ├─ Actions: What actually happened
      ├─ Assessment: What went well/poorly
      └─ Recommendations: What to improve next time
```

## Ejemplos de Aplicación

### Ejemplo 1: Debugging Error de Código

#### OODA Loop:
```
O: Getting "Connection refused" error
O: Network issue, wrong config, or service down
D: Check connection step by step
A: Verify port, test local first, check firewall

O: Local works, remote fails  
O: Likely firewall or network config issue
D: Check network configuration
A: Update firewall rules
→ RESOLVED
```

#### SMART Capture:
```
S: Debugging network connection error
M: Local connections work but remote fail - pattern suggests network layer issue
A: Why do network errors manifest this way? How to debug network vs application vs config issues systematically?
R: Network debugging follows OSI model layers, similar to other system troubleshooting hierarchies
T: Document network debugging checklist, learn more about firewall configuration
```

### Ejemplo 2: Learning New Technology

#### WOOP Planning:
```
W: Learn React hooks for new project
O: Can build functional component with state management  
O: Complex concepts, easy to get overwhelmed, many tutorials with different approaches
P: Start with useState only, build simple counter, then add useEffect
```

#### SMART Learning:
```
S: Learning React useState hook
M: useState returns array [value, setter], functional approach vs class components
A: Why hooks vs classes? How does React track hook calls? What are the rules of hooks?
R: Hooks = functional programming approach, similar to state management in other frameworks
T: Practice with multiple state variables, research useEffect next
```

### Ejemplo 3: Architecture Decision

#### OODA Decision Making:
```
O: Need to choose between REST API vs GraphQL
O: Both have pros/cons, depends on use case requirements
D: Evaluate based on project specific needs: data fetching patterns, team expertise, ecosystem
A: Create simple prototype with both, measure development velocity

O: REST faster to implement, team familiar
O: GraphQL benefits not critical for current scope  
D: Go with REST for MVP, consider GraphQL for v2
A: Implement REST API design
→ DECISION MADE
```

## Tips para Implementación Efectiva

### Para Máxima Eficiencia:
1. **No uses todos los frameworks a la vez** - elige el apropiado para el momento
2. **Mantén notas breves** - frameworks son herramientas, no el objetivo
3. **Adapta a tu estilo** - modifica según tu forma natural de pensar
4. **Practica consistentemente** - hasta que sea automático

### Red Flags - Cuándo NO usar:
- Problemas triviales que no requieren análisis
- Cuando el framework toma más tiempo que la solución
- Si te paraliza en lugar de ayudarte a avanzar

### Para Diferentes Tipos de Problemas:

#### Problemas Técnicos Urgentes:
**Solo OODA** - necesitas acción rápida

#### Aprendizaje Profundo:  
**WOOP + SMART** - planificación + captura

#### Proyectos Complejos:
**Todos los frameworks** en secuencia

#### Debug Sessions:
**OODA (resolver) + SMART (aprender)**

## Adaptación Personal

### Personaliza según tu estilo:
- Si eres visual: Añade diagramas a cada framework
- Si eres rápido: Versiones ultra-cortas de cada uno
- Si eres detallista: Expande cada sección según necesites

### Combina con tus herramientas existentes:
- Notion para tracking WOOP goals
- Libreta para SMART observations
- Digital docs para AAR reviews
- Git commits para OODA action records

## Framework de Meta-Frameworks

Cuando tengas múltiples problemas o proyectos concurrentes:

1. **WOOP** cada nuevo scope/proyecto
2. **OODA** para cada bloqueo específico dentro del scope
3. **SMART** continuo para capturar learning cross-project
4. **AAR** para review de milestones y transfer de knowledge

---

**Esta guía es independiente de tecnología o dominio - funciona para desarrollo de software, investigación académica, aprendizaje de nuevas habilidades, resolución de problemas de negocio, o cualquier desafío estructurado.**