kill -15 `ps ax |grep controller | grep config | awk '{print $1}'`

