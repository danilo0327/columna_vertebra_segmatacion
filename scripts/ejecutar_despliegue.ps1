# Script simplificado para ejecutar despliegue paso a paso
$EC2IP = "100.28.216.213"
$KeyPath = "C:\Users\ASUS\Downloads\final_key.pem"
$User = "ubuntu"

Write-Host "Ejecutando script de construccion en EC2..." -ForegroundColor Yellow
$output = ssh -i $KeyPath $User@$EC2IP "cd /home/ubuntu/columna_vertebra_segmatacion && bash /tmp/build_docker.sh" 2>&1
Write-Host $output

Write-Host "`nVerificando contenedores..." -ForegroundColor Yellow
$containers = ssh -i $KeyPath $User@$EC2IP "sudo docker ps -a" 2>&1
Write-Host $containers

Write-Host "`nVerificando imagenes..." -ForegroundColor Yellow
$images = ssh -i $KeyPath $User@$EC2IP "sudo docker images" 2>&1
Write-Host $images


