# 📘 Guía sencilla de flujo Git y mensajes de commit

Esta guía resume cómo organizar ramas y escribir commits claros siguiendo **Conventional Commits**.

---

## 1️⃣ Ramas principales

- **main** → código estable (producción)
- **dev** → integración de features

---

## 2️⃣ Ramas de trabajo

- `feature/<nombre>` → nueva funcionalidad  
- `fix/<nombre>` → corregir errores  
- `refactor/<nombre>` → reorganizar código  
- `chore/<nombre>` → tareas auxiliares  
- `docs/<nombre>` → documentación  

Ejemplo:
```bash
git checkout dev
git pull origin dev
git checkout -b feature/motion
```

---

## 3️⃣ Mensajes de commit (Conventional Commits)

**Formato**
```
<tipo>(<scope>): <resumen corto en imperativo>

<cuerpo opcional>
<footer opcional>
```

**Tipos más usados**
- `feat` → nueva funcionalidad
- `fix` → corrección de bug
- `refactor` → cambio interno
- `perf` → mejora de rendimiento
- `docs`, `test`, `chore`

**Scope** = módulo o rama (ej: `motion`, `imu`, `navegacion`).

---

## 4️⃣ Ejemplos de commits

### En `feature/motion` (integración IMU)
```
feat(motion): scaffold IMUService y puertos
chore(motion): añadir imu-sdk a requirements
feat(motion): lectura gyro/accel/mag con timestamps
feat(motion): aplicar calibration desde JSON
feat(motion): fusion gyro+accel con heading magnetómetro
perf(motion): vectorizar actualizaciones y usar hilo dedicado
fix(motion): corregir sincronía con RGB/SLAM
test(motion): añadir fixtures y unit tests
docs(motion): README pipeline IMU + diagrama
```

---

## 5️⃣ Flujo recomendado

1. Crear rama `feature/...` desde `dev`  
2. Commits pequeños y claros  
3. Rebase con `dev` antes de abrir PR  
4. **Squash & merge** PR → `dev`  
5. Release de `dev` → `main`

---

## 6️⃣ Buenas prácticas

- Repite el **scope** en el commit aunque esté en el nombre de la rama → el commit debe ser autoexplicativo.  
- Evita mensajes genéricos (`wip`, `update`, `cambios`).  
- Usa el cuerpo del commit para explicar **por qué**, no solo el qué.  
- Borra ramas feature tras el merge.  

---

✅ Con esto tendrás un historial de commits **claro, consistente y fácil de mantener**.
