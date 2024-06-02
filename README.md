# WEB Nhận Diện Kích Thước Tôm 

## I. Tổng Quan Về Trang Web

Trong thời kỳ cách mạng số 4.0, việc áp dụng công nghệ thông tin để tối ưu hóa quy trình sản xuất và quản lý sản phẩm là một xu hướng không thể phủ nhận. Đặc biệt, trong ngành công nghiệp chế biến thực phẩm, việc tự động hóa quy trình nhận dạng, đếm số lượng và phân loại sản phẩm trên băng chuyền là một yếu tố quan trọng để tăng cường hiệu suất và chất lượng sản phẩm. Trong bối cảnh này, việc xây dựng ứng dụng nhận dạng, đếm số lượng và phân loại Tôm trên băng chuyền nhà máy chế mang lại nhiều lợi ích kinh tế cao.
Chúng tôi chọn mô hình mạng học sâu YOLOv8 để triển khai trong dự án này vì tính linh hoạt và độ chính xác cao của nó trong việc nhận dạng và đếm đối tượng trên hình ảnh. Mạng YOLOv8 đã được chứng minh là hiệu quả trong nhiều ứng dụng thị giác máy tính khác nhau và có thể được tinh chỉnh để phù hợp với yêu cầu cụ thể của ngành công nghiệp chế biến thực phẩm.
Lý do chúng tôi chọn đề tài này là để cải thiện quy trình sản xuất tôm và giảm bớt sai sót trong quá trình nhận dạng và phân loại sản phẩm. Việc tự động hóa quy trình này không chỉ giúp tiết kiệm thời gian và nhân lực mà còn đảm bảo tính chính xác và đồng nhất trong sản phẩm cuối cùng. Đồng thời, việc sử dụng công nghệ nhận dạng hình ảnh cũng mở ra tiềm năng ứng dụng trong việc theo dõi và quản lý chất lượng sản phẩm theo thời gian thực.
 	Chúng tôi sẽ tiến hành nghiên cứu và phát triển ứng dụng dựa trên mạng YOLOv8 để nhận dạng, đếm số lượng và phân loại tôm trên băng chuyền nhà máy chế biến. Quá trình này sẽ bao gồm việc thu thập dữ liệu, huấn luyện mô hình, và thử nghiệm trên môi trường thực tế để đảm bảo tính hiệu quả và đáng tin cậy của hệ thống.
Để đạt được mục tiêu này, chúng tôi sẽ tiến hành các thử nghiệm để so sánh kết quả giữa các lần hiệu chỉnh thông số khi huấn luyện trên mô hình. Qua đó, chúng tôi hy vọng sẽ đạt được kết quả tối ưu nhất và có thể triển khai ứng dụng trong môi trường sản xuất thực tế.
Bằng cách này, chúng tôi hi vọng rằng dự án của mình sẽ đóng góp vào sự cải thiện toàn diện của quy trình sản xuất và quản lý trong ngành công nghiệp chế biến thực phẩm, đồng thời khẳng định vai trò quan trọng của công nghệ trong việc nâng cao hiệu suất và chất lượng sản phẩm.

### Công nghệ sử dụng 
Sử dụng các công cụ và môi trường sau đây để xây dựng ứng dụng nhận dạng, đếm số lượng và phân loại tôm trên băng chuyền nhà máy chế biến:
-	Cơ sở dữ liệu: Xây dựng bộ dữ liệu từ hình ảnh các con tôm trên băng chuyền nhà máy chế biến, giúp chuẩn bị dữ liệu huấn luyện cho các mô hình.
-	Công cụ thực thi: Kaggle, makesense.ai được sử dụng trong quá trình phát triển và triển khai dự á. Makesense.ai được sử dụng để gắn nhãn các hình ảnh tôm trên băng chuyền. Kaggle được sử dụng để thực hiện quá trình huấn luyện mô hình, giúp tăng tốc độ và tiết kiệm tài nguyên.
-	Mô hình sử dụng: Sử mô hình YOLOv8 để nhận diện, đếm số lượng và phân loại tôm trên băng chuyền.
Trong tương lai, ứng dụng của hệ thống nhận dạng này có thể được mở rộng để tích hợp vào quy trình sản xuất thực tế, đồng thời cung cấp cái nhìn trực quan và chính xác về chất lượng sản phẩm cho người dùng và quản lý.
-	Sử dụng các ngôn ngữ sau để xây dựng Website:  ngôn ngữ Python cụ thể là Framework Flask, MySQL, HTML, CSS, Bootstrap.

## II. Nội dung Về Dự án

### 1. Huấn luyện mô hình về YoloV8
Các tập dữ liệu hình ảnh về tôm cho mô hình được chụp và thu thập từ nhiều nguồn ảnh khác nhau bao gồm cả hình ảnh có chất lượng tốt rõ nét hay các ảnh bị mờ, nhiễu cùng với các kiểu chụp khác nhau. Tập dữ liệu chứa hình ảnh về tôm được chụp với số lượng tôm nhiều và ít luân phiên, nhiều góc độ và chất lượng ánh sáng màu sắc khác nhau giúp tạo nên sự đa dạng cho tập dữ liệu.
<img src = "https://i.imgur.com/Apo3z5h.png"/>

Tiền xử lý dữ liệu : Truy cập vào website makesense.ai và tạo ba lớp đối tượng cần nhận diện là “BigShrimp”, “MediumShrimp” và  “SmallShrimp” sau đó tiến hành gán nhãn các đối tượng trong ảnh.

<img src = "https://i.imgur.com/g4VmdP6.png"/>
Sau khi xuất tất cả các nhãn đã được gán từ bộ dữ liệu ta sẽ thu được thư mục labels chứa các tập tin txt mà ta đã gán nhãn cho từng ảnh. 
Cài đặt các môi trường và cấu hình cho việc huấn luyện trên YOLOv8: Để bắt đầu quá trình huấn luyện trên YOLOv8, cần phải cài đặt các môi trường và cấu hình phù hợp. Để huấn luyện dữ liệu chúng ta cần sử dụng GPU (Graphics Processing Unit) để tăng tốc độ train, chúng ta có thể sử dụng GPU của Nvidia bằng cài đặt CUDA (Compute Unified Device Architecture)  và Pytorch. Với CUDA là một nền tảng tính toán song song được phát triển bởi Nvidia cho phép người dùng tận dụng khả năng của GPU để tính toán, xử lý các tác vụ phức tạp của học máy, học sâu.


### 2. Tổng quan về Website
#### 2.1 Trang Home
<img src = "https://i.imgur.com/GlnYqOO.png"/>


#### 2.2 Trang Classification
<img src = "https://i.imgur.com/bt3kLFZ.png"/>
 
#### 2.3 Trang History
 <img src = "https://i.imgur.com/J77jTVR.png"/>

#### 2.4 Trang Overview
 <img src = "https://i.imgur.com/RzmNnti.png"/>

#### 2.5 Trang Setting
 <img src = "https://i.imgur.com/DCiTG8n.png"/>

#### 2.6 Trang Login
 <img src = "https://i.imgur.com/tGDwHHi.png"/>
#### 2.7 Trang Register
  <img src = "https://i.imgur.com/JyGw9so.png"/>


### III. Chức năng của Website
Website sẽ được xây dựng bằng python cụ thể là framework flask, trang web sẽ cho phép người dùng đăng tải các hình ảnh có kích thước khác nhau và sẽ vẽ một khung bounding box quanh đối tượng đi kèm tên lớp và độ tin cậy mà mô hình dự đoán và website sẽ bao gồm các chức năng sau:
-	Đăng ký, đăng nhập để bất đầu quá trình nhận dạng.
-	Tải ảnh và video (upload ảnh và video): Tạo nút button cho người dùng tải một bức ảnh hoặc video từ thư mục trong máy lên website và tiến hành dự đoán
-	Kéo thả ảnh, video vào khu vực: Tạo một khu vực cho phép người dùng kéo thả ảnh hoặc video có chứa tôm từ trên mạng mà không cần tải ảnh hoặc video về máy hoặc kéo thả ảnh, video từ thư mục trong máy.
-	Tải ảnh trực tiếp qua URL: Sau khi ảnh đã được tải từ URL, ảnh sẽ được xử lý nhận dạng.
-	Thông kê số liệu tôm đã được nhận dạng từ tổng số ảnh của người dùng.
-	Chỉnh sửa thông tin cá nhân.
  
#### 1. Đăng kí - Đăng nhập tài khoản với Google

<img src = "https://i.imgur.com/EtGnOEP.png"/>

Xử lý hoàn toàn với tài khoản Google của bạn, từ đăng kí đến đăng nhập

#### 2. Tải hình ảnh bằng đường dẫn hoặc từ thiết bị của bạn để xử lý phân loại (CHỨC NĂNG CHÍNH)

<img src = "https://i.imgur.com/MbqKXOm.png"/>


Hình ảnh con tôm ở trên đã được phân loại dựa theo kích thước. Như ảnh trên thì con tôm này thuộc loại tôm lớn. Sau khi nhận diện xong hệ thống sẽ trả về kết quả số lượng con tôm đã được nhận diện bên phần Result và đưa ra số liệu thống kê


<img src = "https://i.imgur.com/XkREqLi.png"/>
Sau khi phân loại từ hình ảnh ở trang History sẽ lưu lại lịch sử và kết quả của các lần xử lý phân loại trước đó và tổng quan thống kê số lượng các loại tôm sau khi phân loại bằng biểu đồ.

<img src = "https://i.imgur.com/U792tvc.png"/>

##### Đây là toàn bộ chức năng chính của Trang Web
##### Mong mọi người có một trải nghiệm vui vẻ với đồ án của tụi mình.... <3


