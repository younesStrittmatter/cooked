#!/bin/bash

# Forzar el uso de punto decimal en printf
LC_NUMERIC="en_US.UTF-8"

# Ruta a tu instalación de conda
CONDA_BASE="/home/samuel_lozano/anaconda3"

# Inicializar conda en bash
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activar entorno
conda activate cooked

# Función para convertir grados a radianes
deg2rad() {
  echo "scale=10; $1 * 4 * a(1) / 180" | bc -l
}

# Lanzar 19 ejecuciones con ángulos de 0 a 90 en pasos de 5 grados
for i in {0..18}; do
  angle=$((i * 5))
  rad=$(deg2rad "$angle")

  # Calcular coseno y seno
  raw_cos_val=$(echo "scale=10; c($rad)" | bc -l)
  raw_sin_val=$(echo "scale=10; s($rad)" | bc -l)

  # Forzar uso del punto decimal (por si la locale usa coma)
  cos_val=$(echo "$raw_cos_val" | tr ',' '.')
  sin_val=$(echo "$raw_sin_val" | tr ',' '.')

  # Agregar 0 antes del punto si falta
  cos_val=$(echo "$cos_val" | sed 's/^\./0./')
  sin_val=$(echo "$sin_val" | sed 's/^\./0./')

  # Validación de número y formateo a 4 decimales
  if [[ $cos_val =~ ^-?[0-9]+(\.[0-9]+)?$ && $sin_val =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
    cos_val=$(LC_NUMERIC="en_US.UTF-8" printf "%.4f" "$cos_val")
    sin_val=$(LC_NUMERIC="en_US.UTF-8" printf "%.4f" "$sin_val")
  else
    echo "Error: valor no numérico detectado (alpha=$angle°): cos='$cos_val', sin='$sin_val'"
    exit 1
  fi

  # Nombres de archivo
  input_file="input_$(printf "%02d" $i).txt"
  output_file="output_$(printf "%02d" $i).txt"

  # Crear archivo de entrada
  cat <<EOF > "$input_file"
PPO
1
2000
5
0.7071
0.7071
$cos_val
$sin_val
EOF

  # Ejecutar el entrenamiento en segundo plano con nohup
  nohup python training-DTDE-spoiled_broth.py < "$input_file" > "$output_file" 2>&1 &

  echo "Lanzado entrenamiento con alpha=${angle}° -> $input_file"
done