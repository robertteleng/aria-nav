# Problem-Solving con 4 Frameworks

## Descripción General

Esta guía proporciona un sistema completo de frameworks mentales para abordar cualquier problema técnico, proyecto de desarrollo, o desafío de aprendizaje de manera sistemática y eficiente.

## Los 4 Frameworks Core

### 1. WOOP - Session Planning
**Propósito:** Planificación y setup de sesiones de trabajo

- **W** - **Wish**: ¿Qué quiero lograr?
- **O** - **Outcome**: ¿Cómo sabré que lo conseguí?
- **O** - **Obstacles**: ¿Qué puede salir mal?
- **P** - **Plan**: ¿Cuál es mi approach específico?

### 2. LOG - Session Notes (Durante el trabajo)
**Propósito:** Captura rápida y no intrusiva durante la ejecución

- **L** - **Line**: Una línea por acción/evento con timestamp
- **O** - **Obstacles**: Marca bloqueos cuando aparecen (!)
- **G** - **Get insights**: Captura aprendizajes rápidos (💡)

### 3. OODA Loop - Problem Solving (Solo cuando hay bloqueos)
**Propósito:** Resolución de problemas específicos cuando el trabajo normal se bloquea

- **O** - **Observe**: ¿Qué está pasando exactamente?
- **O** - **Orient**: ¿Cómo se relaciona con lo que sé?
- **D** - **Decide**: ¿Cuál es la mejor acción?
- **A** - **Act**: Ejecutar y medir resultado

### 4. AAR - Session Review & Learning Capture
**Propósito:** Revisión, mejora continua y captura de aprendizajes

- **A** - **Actions**: ¿Qué pasó realmente?
- **A** - **Assessment**: ¿Qué fue bien/mal? + ¿Qué aprendí?
- **R** - **Recommendations**: ¿Qué cambiar próxima vez? + ¿Qué insights aplicar?

## Cuándo Usar Cada Framework

### WOOP - Al Inicio de Cualquier Sesión
```
✓ Empezar nuevo proyecto
✓ Comenzar nueva funcionalidad  
✓ Abordar problema complejo
✓ Sesión de aprendizaje enfocada
✓ Cuando necesitas claridad de objetivos
```

### LOG - Durante Todo el Trabajo (Continuo)
```
✓ Registrar progreso normal con timestamps
✓ Marcar cuando aparecen bloqueos  
✓ Capturar insights mientras trabajas
✓ Mantener historial de lo que haces
✓ Base de datos para AAR posterior
```

### OODA Loop - Solo Cuando Hay Bloqueos
```
✓ Error que no entiendes
✓ Decisión técnica compleja
✓ Stuck por más de 10-15 minutos
✓ Multiple soluciones posibles, no sabes cuál elegir
✓ Debugging de problema no obvio
✓ Necesitas cambiar approach fundamentalmente
```

### AAR - Final de Sesión/Milestone
```
✓ Completar funcionalidad importante
✓ Resolver problema mayor
✓ Final del día de trabajo  
✓ Milestone del proyecto alcanzado
✓ Después de aprender algo significativo
```

## Framework Integration Workflow

### Flujo Completo con LOG

```
START SESSION:
│
├─ WOOP (Planning)
│  ├─ Define wish/outcome for session
│  ├─ Identify potential obstacles  
│  └─ Create specific action plan
│
├─ WORK WITH LOG:
│  │
│  ├─ LOG - Continuous capture:
│  │  ├─ Line: Record what you're doing with timestamps
│  │  ├─ Obstacles: Mark blocks when they appear (!)
│  │  ├─ Get insights: Quick learning notes (💡)
│  │  └─ Keep working normally
│  │
│  └─ IF BLOCKED → Use OODA:
│     ├─ Observe what's happening
│     ├─ Orient to existing knowledge  
│     ├─ Decide on approach
│     ├─ Act and measure result
│     ├─ LOG the resolution
│     └─ Return to normal work
│
└─ END SESSION:
   │
   └─ AAR (Review + Learning)
      ├─ Use your LOG notes as reference
      ├─ Actions: What actually happened
      ├─ Assessment: What went well/poorly + What did I learn?
      └─ Recommendations: What to change + What insights to apply?
```

## La Clave: Trabajo Normal vs Problem-Solving Mode

### Trabajo Normal (80-90% del tiempo)
- **Flow state**: Implementas, refactorizas, documentas
- **Decisiones rutinarias**: Nombres de variables, estructura de archivos
- **Progreso incremental**: Cada paso es lógico y claro
- **No frameworks**: Solo ejecutas tu plan de WOOP

### Problem-Solving Mode (10-20% del tiempo)  
- **Bloqueos reales**: No sabes cómo proceder
- **Decisiones complejas**: Multiple trade-offs a considerar
- **Debugging difícil**: El error no es obvio
- **Usa OODA**: Análisis sistemático necesario

## Ejemplos de Aplicación

### Ejemplo 1: Desarrollo de Feature Nueva

#### WOOP Planning:
```
W: Implementar sistema de comentarios
O: Usuarios pueden crear, editar, eliminar comentarios exitosamente
O: UI complex, database design decisions, real-time updates challenging
P: Start with basic CRUD, simple UI, worry about real-time later
```

#### LOG durante el trabajo:
```
[25/08 - Comments Feature]

GOAL: Basic CRUD for comments

10:30 → Database model creation
10:45 → API endpoints setup  
11:15 ! OAuth callback issue → OODA needed
11:45 → Back to API, OAuth fixed
💡 OAuth URLs must match exactly - env configs important
12:00 → Frontend component started
12:30 → Basic form working
💡 State management trickier than expected, context vs props
13:00 → Testing locally ✓

NEXT: Connect frontend to backend
```

#### OODA (cuando aparece el bloqueo):
```
BLOQUEO: ¿Cómo manejar comentarios anidados (replies) en la UI?

O: Flat list doesn't show relationships, nested structure could be complex to render
O: Other apps use threading (Reddit) or flat with visual indicators (Twitter)
D: Start with simple threading, max 2 levels deep to avoid complexity
A: Implement recursive component with depth limit

→ LOG: 11:45 → Threading solution implemented ✓
→ Unblocked, back to normal work
```

#### AAR Review (usando LOG como referencia):
```
A: Implemented basic commenting system, handled threading challenge
A: Went well: API design was clean, database model worked, OODA helped with threading decision
   Learned: Threading UI trickier than expected, recursive components need careful state management
   Learned: OAuth configuration is environment-sensitive, document early
   Went poorly: Could have planned frontend state management better
R: Next time: Design state management before UI components
   Apply: Pattern of recursive components useful for other tree-like data
   Apply: Environment-specific config checklist for OAuth integrations
```

### Ejemplo 2: Bug Investigation

#### WOOP Planning:
```
W: Fix performance issue - page loads slowly
O: Page loads in under 2 seconds consistently
O: Could be database, network, frontend rendering, or caching issue  
P: Use browser dev tools first, then profile step by step
```

#### LOG durante investigación:
```
[25/08 - Performance Bug]

GOAL: Page load under 2 seconds

14:30 → Open dev tools, check network tab
14:40 → Database query logs review
💡 Individual metrics look normal
15:00 → Server response times check
15:15 ! Still slow but metrics look good → OODA needed
15:45 → Found JS bundle issue via profiling
💡 Bundle size = sneaky performance killer
16:00 → Code splitting implemented ✓
16:15 → Performance testing - under 2 seconds ✓

KEY INSIGHT: Always profile before optimizing
```

#### OODA (cuando no es obvio):
```
BLOQUEO: All metrics look normal individually, but page still slow

O: Network fast, DB queries fast, but total page load 5+ seconds
O: Could be waterfall loading, blocking resources, or client-side processing
D: Profile client-side JavaScript execution and resource loading order
A: Use Chrome DevTools Performance tab to trace execution

O: Found it! Large JavaScript bundle blocking initial render
O: Bundle includes unused libraries, no code splitting
D: Implement code splitting and remove unused dependencies
A: Configure webpack code splitting, audit dependencies

→ LOG: 15:45 → Bundle optimization complete ✓
→ Problem solved, performance improved
```

#### AAR Review (con LOG como base):
```
A: Fixed performance issue by optimizing JavaScript bundle size
A: Went well: Systematic approach with LOG helped track investigation steps
   Learned: Performance issues often aren't where you first look, profiling tools essential
   Learned: Bundle size can be sneaky performance killer even with fast network
   Learned: LOG helped me see the investigation timeline clearly
R: Next time: Check bundle size early in development, set up performance monitoring
   Apply: Always profile before optimizing, measurement beats guessing
   Apply: LOG pattern useful for debugging - shows investigation flow clearly
```

## Formato LOG para Notas a Mano

### Template Básico:
```
[Fecha - Proyecto/Feature]

GOAL: [Tu WOOP outcome en 1 línea]

[Timestamp] → [Action/Task]
[Timestamp] ! [Bloqueo description] → OODA
[Timestamp] → [Resolution] ✓  
💡 [Quick insight]
[Timestamp] → [Next action]

NEXT: [Immediate next steps]
```

### Sistema de Símbolos LOG:
```
→ Normal work/progress (Line)
! Problem/bloqueo (Obstacles) 
💡 Insight/learning (Get insights)
✓ Completed task
? Decision pending  
⚠ Watch out for this
```

### Ejemplo Real de Notas:
```
[01/09 - User Auth]

GOAL: OAuth login working end-to-end

09:30 → Setup OAuth client credentials
10:00 → Login button component  
10:30 ! Redirect URI mismatch → OODA
💡 Dev vs prod URLs need different configs
11:00 → Environment configs updated ✓
11:15 → Testing locally ✓
💡 OAuth callback timing is tricky
11:30 → Deploy to staging
11:45 → End-to-end test successful ✓

NEXT: Production deployment checklist
```

## Tips para LOG Efectivo

### Durante el Trabajo:
1. **Timestamps cada 15-30 min** - No cada minuto
2. **Una línea por evento** - Mantén brevedad  
3. **Marca bloqueos inmediatamente** - "!" cuando aparecen
4. **Insights al momento** - "💡" cuando los tengas
5. **No te detengas mucho** - LOG no debe interrumpir flow

### Para AAR Después:
- **LOG es tu memoria externa** - Revísalo antes de AAR
- **Patrones emergen** - ¿Dónde te bloqueas más?
- **Timeline real** - ¿Cuánto tardó realmente cada cosa?
- **Insights documentation** - Los 💡 se vuelven knowledge base

## Cuándo NO Usar Frameworks

### Skip LOG para:
- Sesiones muy cortas (menos de 30 min)
- Trabajo puramente rutinario sin decisiones
- Cuando estás en deep flow y interrumpe

### Skip OODA para:
- Decisiones simples (¿cómo nombrar esta variable?)
- Errores obvios (typo en código, missing import)  
- Trabajo rutinario que ya sabes hacer
- Cuando el "bloqueo" es solo necesidad de Google algo rápido

### Skip AAR para:
- Tareas muy pequeñas (fix typo, update README)
- Sesiones de menos de 45 minutos
- Trabajo puramente rutinario sin insights

### Nunca Skip WOOP:
- Siempre útil para cualquier sesión de trabajo significativa
- Incluso 2 minutos de planning ahorra tiempo después

## Implementación Gradual de los 4 Frameworks

### Semana 1: Solo WOOP
- Practica 2 minutos de planning al inicio de cada sesión
- No uses otros frameworks aún
- Acostúmbrate a definir outcome claro

### Semana 2: WOOP + LOG  
- Añade notas simples durante el trabajo
- Solo timestamps y acciones básicas (→)
- No te preocupes por perfección

### Semana 3: WOOP + LOG + OODA
- Cuando tengas bloqueo real, usa OODA conscientemente  
- Marca los bloqueos en LOG (!)
- Ve la conexión entre LOG y OODA

### Semana 4: Sistema Completo
- Añade AAR al final de sesiones importantes
- Usa LOG como base para AAR
- Optimiza según tu experiencia

### Señales para Usar OODA:
- Llevas 10+ minutos sin progresar
- Has probado 2-3 approaches sin éxito
- No estás seguro cuál de varias opciones elegir
- El error/problema no tiene causa obvia
- Necesitas research significativo para decidir

### Personalización por Contexto:

#### Para Desarrollo Web:
- **WOOP**: Include performance/accessibility goals
- **OODA**: Common for debugging, API integration, responsive design
- **AAR**: Focus on UX learnings, code quality insights

#### Para Data Science:
- **WOOP**: Define success metrics clearly  
- **OODA**: Model selection, feature engineering decisions
- **AAR**: Document model insights, data quality learnings

#### Para DevOps/Infrastructure:
- **WOOP**: Include rollback plan in obstacles
- **OODA**: Troubleshooting system issues, capacity planning  
- **AAR**: Incident learnings, automation opportunities

## Adaptación Personal

### Ajusta según tu estilo de trabajo:
- **Short sessions**: WOOP verbal (2 min), skip AAR unless significant learning
- **Long sessions**: Full written WOOP, detailed AAR with examples
- **Pair programming**: OODA collaborative, shared AAR insights  
- **Solo deep work**: Minimal interruption, OODA only for real blocks

### Combina con herramientas existentes:
- **Notion/Obsidian**: Template pages for WOOP and AAR
- **Git**: Use commit messages to capture OODA decisions  
- **Calendar**: Block time for AAR review at end of work days
- **Notes app**: Quick OODA capture during problem-solving

---

**Esta guía con 4 frameworks integrados crea un sistema completo: WOOP (planning) → LOG (tracking) → OODA (problem-solving) → AAR (review). Cada framework tiene su momento específico y juntos forman un ciclo de mejora continua.**