void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(13,OUTPUT);
}

void playBitcode(int port, int n) {
  /* 
   *  n should not be larger than 128. 
   */

  const int WIDTH = 10;
  const int HEAD_INTERVAL = 100;
  const int CODE_INTERVAL = 50;
  const int LEN_CODE = 8;

  // play head
  digitalWrite(port, HIGH);
  delay(WIDTH);
  digitalWrite(port, LOW);
  delay(HEAD_INTERVAL);
   
  // play bitcode
  int i = 0;
  int res = 0;
  while (i < LEN_CODE) {
    res = n % 2;
    if (res == 0) {
      n = n / 2;
      digitalWrite(port, LOW);
    } else {
      n = (n-1) / 2;
      digitalWrite(port, HIGH);
    }
    delay(WIDTH);
    digitalWrite(port, LOW);
    delay(CODE_INTERVAL);
    i++;
  }
}

void loop() {
  const int MAX_SIG = 128;
  int sig = 1;

  delay(10000);

  while (sig < MAX_SIG) {
    playBitcode(13, sig);
    delay(1000);
    sig++;
  }
}
