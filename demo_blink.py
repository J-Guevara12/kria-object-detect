from pynq import Overlay
from time import sleep

# 1. Cargar nuestro Hardware personalizado
# PYNQ buscará automáticamente el .hwh que tenga el mismo nombre
ol = Overlay("blink_design.bit")

# 2. Localizar el driver del GPIO
# Podemos acceder por el nombre que le dio Vivado o por la clase
# 'axi_gpio_0' es el nombre por defecto del bloque en Vivado
led_ip = ol.axi_gpio_0

# 3. Configurar el canal
# El AXI GPIO tiene dos canales, usamos el canal 1 por defecto.
# .write(offset, value). El offset de datos es 0x00 para el canal 1.
# O usamos la abstracción de PYNQ para leer/escribir más fácil.

print("Iniciando parpadeo del LED UF1...")

try:
    while True:
        # Encender (Escribir 1 en el registro de datos)
        led_ip.write(0x0, 0x1) 
        sleep(0.5)
        
        # Apagar (Escribir 0 en el registro de datos)
        led_ip.write(0x0, 0x0)
        sleep(0.5)
        
except KeyboardInterrupt:
    print("Parpadeo detenido.")
    # Asegurarse de apagar el LED al salir
    led_ip.write(0x0, 0x0)
