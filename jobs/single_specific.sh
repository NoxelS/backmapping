sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_b47.log --error=./jobs/logs/single_b47.err --wrap="jobs/train_single.sh 47"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_b48.log --error=./jobs/logs/single_b48.err --wrap="jobs/train_single.sh 48"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_b49.log --error=./jobs/logs/single_b49.err --wrap="jobs/train_single.sh 49"

sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_a55.log --error=./jobs/logs/single_a55.err --wrap="jobs/train_single.sh 55"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_a56.log --error=./jobs/logs/single_a56.err --wrap="jobs/train_single.sh 56"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_a57.log --error=./jobs/logs/single_a57.err --wrap="jobs/train_single.sh 57"

sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_d140.log --error=./jobs/logs/single_d140.err --wrap="jobs/train_single.sh 140"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_d141.log --error=./jobs/logs/single_d141.err --wrap="jobs/train_single.sh 141"
sbatch --exclude=fang1,fang48,fang51,fang52,fang53,fang54 --job-name=TRAIN --gres=gpu:1 --mem-per-gpu=11G --nodes=1 --output=./jobs/logs/single_d142.log --error=./jobs/logs/single_d142.err --wrap="jobs/train_single.sh 142"
