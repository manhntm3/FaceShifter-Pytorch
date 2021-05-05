## From hoian to host
# rsync -aP --exclude 'FaceShifter/checkpoints' --exclude 'FaceShifter/.git' --exclude '*.pth' ubuntu@hoian:/home/ubuntu/FaceShifter /home/manh/

## From host to tessera

# rsync -aP --exclude 'FaceShifter/checkpoints' --exclude 'FaceShifter/.git' --exclude '*.pth' /home/manh/FaceShifter root@tessera:/root/ 
rsync -aP --exclude 'FaceShifter/checkpoints' --exclude 'FaceShifter/.git' --exclude '*.pth' root@tessera:/root/FaceShifter /home/manh/

## From host to lyson

# rsync -aP --exclude 'FaceShifter/checkpoints' --exclude 'FaceShifter/.git' --exclude '*.pth' /home/manh/FaceShifter ubuntu@lyson:/home/ubuntu/ 
