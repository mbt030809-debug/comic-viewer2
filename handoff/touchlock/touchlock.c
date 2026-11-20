// touchlock: Claude가 adb로 폰을 조작하는 동안 사용자 터치만 막는다.
// 원리: 터치스크린 장치를 EVIOCGRAB으로 독점 → 안드로이드는 실제 터치를 못 받음.
//       adb `input tap/swipe`는 커널을 거치지 않고 InputDispatcher로 주입되므로 그대로 동작.
// 정지: (1) 화면의 빨간 정지 버튼(오버레이 앱이 위치를 알려 줌; 앱이 없으면 오른쪽 위 모서리)
//       (2) 볼륨 아래 키  (3) 시간 초과  (4) adb 끊김(SIGHUP)
// 오버레이 앱(com.example.touchlock)은 127.0.0.1:38999로 접속해 "R 회전 폭 높이 x0 y0 x1 y1" 줄을 보낸다.
// 이 프로그램이 끝나면 연결이 끊기고 앱은 스스로 사라진다.
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static const char *touch_devs[] = {"sec_touchscreen", "sec_touchscreen2", "sec_touchpad"};
#define NT 3
static int fds[8], nfd = 0, keyfd = -1;
static int cx[8], cy[8];
static volatile sig_atomic_t quit = 0;
static int lsock = -1, csock = -1;
static int have_rect = 0, rot, sw, sh, bx0, by0, bx1, by1;
static char lbuf[256]; static int llen = 0;

// 원시 좌표(0..4095, 패널 기본 방향) → 현재 화면 좌표 → 버튼 안인가
static int in_button(int rx, int ry) {
  if (!have_rect) return rx > 3600 && ry >= 0 && ry < 500;  // 앱 없을 때: 오른쪽 위 모서리
  int nw = (rot & 1) ? sh : sw, nh = (rot & 1) ? sw : sh;
  int px = rx * nw / 4095, py = ry * nh / 4095, lx, ly;
  switch (rot) {
    case 1: lx = py; ly = nw - px; break;
    case 2: lx = nw - px; ly = nh - py; break;
    case 3: lx = nh - py; ly = px; break;
    default: lx = px; ly = py;
  }
  int pad = 24;
  return lx >= bx0 - pad && lx <= bx1 + pad && ly >= by0 - pad && ly <= by1 + pad;
}

static void read_client(void) {
  int n = read(csock, lbuf + llen, sizeof lbuf - 1 - llen);
  if (n <= 0) { close(csock); csock = -1; llen = 0; return; }
  llen += n; lbuf[llen] = 0;
  char *nl;
  while ((nl = strchr(lbuf, '\n'))) {
    *nl = 0;
    int a[7];
    if (sscanf(lbuf, "R %d %d %d %d %d %d %d", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]) == 7) {
      rot = a[0]; sw = a[1]; sh = a[2]; bx0 = a[3]; by0 = a[4]; bx1 = a[5]; by1 = a[6]; have_rect = 1;
      printf("button rot=%d screen=%dx%d rect=%d,%d-%d,%d\n", rot, sw, sh, bx0, by0, bx1, by1); fflush(stdout);
    }
    int rest = llen - (int)(nl + 1 - lbuf);
    memmove(lbuf, nl + 1, rest); llen = rest; lbuf[llen] = 0;
  }
  if (llen >= (int)sizeof lbuf - 1) llen = 0;
}
static void on_sig(int s) { (void)s; quit = 1; }

static int open_by_name(const char *want) {
  char path[64], name[128];
  for (int i = 0; i < 32; i++) {
    snprintf(path, sizeof path, "/dev/input/event%d", i);
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) continue;
    name[0] = 0; ioctl(fd, EVIOCGNAME(sizeof name), name);
    if (strcmp(name, want) == 0) return fd;
    close(fd);
  }
  return -1;
}

int main(int argc, char **argv) {
  int timeout = argc > 1 ? atoi(argv[1]) : 600;  // 기본 10분 후 자동 해제
  signal(SIGHUP, on_sig); signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
  for (int i = 0; i < NT; i++) {
    int fd = open_by_name(touch_devs[i]);
    if (fd < 0) continue;
    if (ioctl(fd, EVIOCGRAB, 1) != 0) { perror(touch_devs[i]); close(fd); continue; }
    cx[nfd] = cy[nfd] = -1; fds[nfd++] = fd;
  }
  keyfd = open_by_name("gpio-keys");
  if (nfd == 0) { fprintf(stderr, "no touch device grabbed\n"); return 1; }
  unlink("/data/local/tmp/touchlock.stopped");
  signal(SIGPIPE, SIG_IGN);
  lsock = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in sa = {0}; sa.sin_family = AF_INET; sa.sin_port = htons(38999); sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(lsock, (struct sockaddr *)&sa, sizeof sa) != 0 || listen(lsock, 2) != 0) { perror("listen"); close(lsock); lsock = -1; }
  printf("LOCKED (%d devices, timeout %ds)\n", nfd, timeout); fflush(stdout);
  time_t end = time(NULL) + timeout;
  const char *why = "timeout";
  while (!quit && time(NULL) < end) {
    struct pollfd p[11]; int n = 0;
    for (int i = 0; i < nfd; i++) p[n++] = (struct pollfd){fds[i], POLLIN, 0};
    if (keyfd >= 0) p[n++] = (struct pollfd){keyfd, POLLIN, 0};
    int li = -1, ci = -1;
    if (lsock >= 0) { li = n; p[n++] = (struct pollfd){lsock, POLLIN, 0}; }
    if (csock >= 0) { ci = n; p[n++] = (struct pollfd){csock, POLLIN, 0}; }
    if (poll(p, n, 1000) <= 0) continue;
    if (li >= 0 && (p[li].revents & POLLIN)) {
      int c = accept(lsock, NULL, NULL);
      if (c >= 0) { if (csock >= 0) close(csock); csock = c; llen = 0; }
    } else if (ci >= 0 && (p[ci].revents & (POLLIN | POLLHUP | POLLERR))) read_client();
    struct input_event ev;
    for (int i = 0; i < nfd; i++)
      while (read(fds[i], &ev, sizeof ev) == sizeof ev) {
        if (ev.type == EV_ABS && ev.code == ABS_MT_POSITION_X) cx[i] = ev.value;
        if (ev.type == EV_ABS && ev.code == ABS_MT_POSITION_Y) cy[i] = ev.value;
        if (ev.type == EV_SYN && cx[i] >= 0 && cy[i] >= 0 && in_button(cx[i], cy[i])) { why = "button"; quit = 1; }
      }
    if (keyfd >= 0)
      while (read(keyfd, &ev, sizeof ev) == sizeof ev)
        if (ev.type == EV_KEY && ev.code == KEY_VOLUMEDOWN && ev.value == 1) { why = "volume"; quit = 1; }
  }
  if (quit && strcmp(why, "timeout") == 0) why = "signal";
  for (int i = 0; i < nfd; i++) { ioctl(fds[i], EVIOCGRAB, 0); close(fds[i]); }
  if (strcmp(why, "button") == 0 || strcmp(why, "volume") == 0) {
    FILE *f = fopen("/data/local/tmp/touchlock.stopped", "w"); if (f) { fputs(why, f); fclose(f); }
  }
  if (csock >= 0) close(csock);
  if (lsock >= 0) close(lsock);
  printf("UNLOCKED (%s)\n", why);
  return 0;
}
