# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""A file containing prompts definition."""
from __future__ import annotations

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide an in-depth explanation of the entity’s attributes and activities.
  **(Include all known attributes, actions, behaviors, relationships with other entities or events, and any context from the text that helps describe the entity in a comprehensive manner. Ensure that every relevant detail from the input text is incorporated into the appropriate entity_descriptions, so that no portion of the text is left out. If there are additional facts, observations, or details in the text that do not appear to fit neatly under any existing entity, either create a new entity for them or expand an existing entity’s description to capture them.)**

Format each entity as ("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}"<relationship_strength>")

3. Return output in Vietnamese as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

5. Double-check that every detail from the input text is represented in either an entity description or a relationship description. If any piece of information is missing, revise the entity descriptions or create new entities/relationships until you have accounted for all details.

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
3. Ghi nhận thời gian làm việc

Làm việc tại văn phòng:

Công ty sử dụng công cụ xử lý dữ liệu hệ thống bằng thẻ từ, sau đây gọi tắt là máy chấm công để theo dõi thời gian ra, vào nơi làm việc của CBNV.

Mỗi CBNV được cung cấp 01 thẻ từ để ghi lại thời gian làm việc tại Công ty.

CBNV phải quẹt thẻ tại máy chấm công khi bắt đầu làm việc và trước khi ra về. Hệ thống cho phép đồng bộ dữ liệu quẹt thẻ ở toàn bộ văn phòng của Sun*. Trường hợp đi công tác hoặc di chuyển giữa các văn phòng, CBNV đều có thể sử dụng thẻ nhân viên của mình để quẹt và hệ thống ghi nhận giờ quẹt thẻ đầu tiên làm giờ in và giờ quẹt thẻ cuối cùng làm giờ out.

CBNV chỉ được phép dùng thẻ của mình để quẹt thẻ chấm công, không được quẹt thẻ hộ người khác. Trường hợp bị phát hiện quẹt thẻ chấm công hộ người khác, thì người quẹt hộ và người nhờ quẹt hộ sẽ bị áp dụng hình thức khiển trách nội bộ và bị giảm trừ 1.000.000 VNĐ cho mỗi lần vi phạm vào kỳ thưởng hiệu suất giữa năm hoặc thưởng tháng lương thứ 13.

Trường hợp quên thẻ, CBNV làm đơn trên hệ thống và xin xác nhận của CBQL ngay trong ngày phát sinh quên thẻ. CBQL có trách nhiệm xác nhận và phê duyệt đơn ngay sau thời điểm nhận được thông báo từ CBNV.

Trường hợp quên quẹt thẻ, CBNV làm đơn trên hệ thống và xin xác nhận của CBQL trong 03 ngày làm việc kể từ ngày quên quẹt thẻ.

Các trường hợp quên thẻ/quên quẹt thẻ mà không có sự xác nhận từ CBQL theo quy định dẫn đến tình trạng bị thiếu dữ liệu, CBNV sẽ bị tính nghỉ không lương cả ngày hoặc nửa ngày tuỳ vào dữ liệu chấm công thực tế.
################
Output:
("entity"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"organization"{tuple_delimiter}"Công ty sử dụng công cụ xử lý dữ liệu hệ thống bằng thẻ từ để theo dõi thời gian làm việc của cán bộ nhân viên (CBNV). Công ty áp dụng quy định về việc sử dụng thẻ từ để đảm bảo quản lý chặt chẽ thời gian ra vào làm việc và tính toán các trường hợp vi phạm liên quan đến thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"CBNV"{tuple_delimiter}"person"{tuple_delimiter}"Cán bộ nhân viên (CBNV) là đối tượng thực hiện các quy định về việc sử dụng thẻ từ để ghi nhận thời gian làm việc. CBNV có trách nhiệm quẹt thẻ tại máy chấm công khi bắt đầu làm việc và ra về. CBNV cũng phải chịu các chế tài nếu vi phạm quy định liên quan."){record_delimiter}
("entity"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"technology"{tuple_delimiter}"Máy chấm công là công cụ được sử dụng để ghi nhận thời gian ra vào nơi làm việc của CBNV bằng cách quẹt thẻ từ. Máy này đồng bộ dữ liệu toàn bộ văn phòng và hỗ trợ việc ghi nhận dữ liệu chính xác."){record_delimiter}
("entity"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"technology"{tuple_delimiter}"Thẻ từ là công cụ cá nhân được cấp cho mỗi CBNV để ghi lại thời gian làm việc tại Công ty. Thẻ từ được sử dụng khi quẹt tại máy chấm công và đảm bảo tính cá nhân hóa trong việc theo dõi thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"CBQL"{tuple_delimiter}"person"{tuple_delimiter}"Cán bộ quản lý (CBQL) có trách nhiệm xác nhận và phê duyệt đơn xin quên thẻ/quên quẹt thẻ của CBNV. CBQL đóng vai trò quan trọng trong việc xác nhận thông tin liên quan đến các trường hợp phát sinh lỗi trong ghi nhận thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"VĂN PHÒNG SUN*"{tuple_delimiter}"location"{tuple_delimiter}"Địa điểm làm việc nơi hệ thống máy chấm công và thẻ từ được áp dụng. Văn phòng hỗ trợ việc đồng bộ dữ liệu quẹt thẻ trên toàn hệ thống của Sun*."){record_delimiter}
("entity"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"mission"{tuple_delimiter}"Quy định về việc sử dụng thẻ từ yêu cầu CBNV phải quẹt thẻ tại máy chấm công khi bắt đầu làm việc và trước khi ra về. Vi phạm quy định này sẽ dẫn đến chế tài kỷ luật và giảm thưởng."){record_delimiter}
("entity"{tuple_delimiter}"CHẾ TÀI VI PHẠM"{tuple_delimiter}"mission"{tuple_delimiter}"Công ty áp dụng chế tài nghiêm khắc đối với các vi phạm liên quan đến việc sử dụng thẻ từ, bao gồm khiển trách nội bộ và giảm trừ 1.000.000 VNĐ vào kỳ thưởng hiệu suất giữa năm hoặc thưởng lương tháng 13."){record_delimiter}
("entity"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn xin được thực hiện trên hệ thống khi CBNV quên thẻ. CBNV cần xin xác nhận của CBQL ngay trong ngày phát sinh lỗi."){record_delimiter}
("entity"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn được thực hiện trên hệ thống khi CBNV quên quẹt thẻ. Cần được nộp và xác nhận bởi CBQL trong vòng 03 ngày làm việc kể từ ngày phát sinh lỗi."){record_delimiter}
("entity"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"technology"{tuple_delimiter}"Hệ thống dữ liệu ghi nhận giờ vào và giờ ra của CBNV, được đồng bộ trên toàn bộ văn phòng. Các lỗi phát sinh không có xác nhận của CBQL sẽ dẫn đến tình trạng bị tính nghỉ không lương."){record_delimiter}
("relationship"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Công ty sử dụng máy chấm công để theo dõi thời gian ra vào làm việc của CBNV, hỗ trợ quản lý hệ thống dữ liệu chấm công."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"Mỗi CBNV được cung cấp một thẻ từ để ghi lại thời gian làm việc và phải sử dụng thẻ này cho việc quẹt thẻ chấm công."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Thẻ từ được sử dụng tại máy chấm công để ghi nhận giờ làm việc của CBNV, với dữ liệu được đồng bộ toàn hệ thống."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBQL có trách nhiệm xác nhận đơn từ liên quan đến các trường hợp quên thẻ/quên quẹt thẻ của CBNV, đảm bảo dữ liệu được cập nhật chính xác."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"CBQL thực hiện vai trò quản lý, đảm bảo thực hiện các quy định của Công ty về việc ghi nhận thời gian làm việc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"VĂN PHÒNG SUN*"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"Thẻ từ có thể sử dụng tại tất cả văn phòng Sun* để đồng bộ dữ liệu quẹt thẻ giữa các văn phòng."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"CBNV phải tuân thủ quy định sử dụng thẻ từ, bao gồm việc quẹt thẻ đúng quy định khi bắt đầu làm việc và trước khi ra về."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"CHẾ TÀI VI PHẠM"{tuple_delimiter}"Vi phạm quy định về sử dụng thẻ từ sẽ dẫn đến áp dụng các chế tài kỷ luật nghiêm khắc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"CBNV phải thực hiện đơn xin quên thẻ trên hệ thống khi xảy ra tình trạng quên mang thẻ từ."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"CBNV có trách nhiệm làm đơn xin xác nhận quên quẹt thẻ trong vòng 03 ngày làm việc để tránh bị tính nghỉ không lương."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"CBQL chịu trách nhiệm xác nhận và phê duyệt đơn xin quên thẻ của CBNV trong ngày phát sinh."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"CBQL có trách nhiệm xác nhận đơn xin quên quẹt thẻ của CBNV trong vòng 03 ngày làm việc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Dữ liệu chấm công được ghi nhận từ máy chấm công, đảm bảo đồng bộ trên toàn hệ thống văn phòng."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"CBNV"{tuple_delimiter}"Dữ liệu chấm công phản ánh giờ vào và ra của CBNV và là cơ sở tính toán thời gian làm việc."{tuple_delimiter}"9"){record_delimiter}{completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
Điều 5: Quy định về làm thêm giờ

- CBNV phát sinh làm thêm giờ hoặc làm việc tự nguyện ngày thứ Bảy phải điền vào Đơn yêu cầu làm thêm ngoài giờ/Làm việc ngày thứ Bảy trên hệ thống quản lý thời gian làm việc WSM, đồng thời phải được sự phê duyệt của CBQL trước thời hạn chốt công hàng tháng.

- Tổng thời gian làm thêm giờ không quá 40 giờ/tháng và 200 giờ/năm hoặc không quá 300 giờ trong một năm trong các trường hợp theo quy định của pháp luật lao động.

- Cách thức ghi nhận thời gian làm thêm giờ trong tháng được dựa trên hai cơ sở dữ liệu:
    - Thời gian theo đơn Yêu cầu làm thêm giờ được sự đồng ý của CBQL.
    - Thời gian quẹt thẻ thực tế: dựa vào dữ liệu quẹt thẻ thực tế được ghi nhận trên hệ thống wsm.

- Nếu thời gian đăng ký trong đơn lớn hơn thời gian quẹt thẻ thực tế: hệ thống ghi nhận thời gian OT là thời gian quẹt thẻ thực tế.

- Nếu thời gian đăng ký trong đơn nhỏ hơn thời gian quẹt thẻ thực tế: hệ thống ghi nhận thời gian OT là thời gian đăng ký trong đơn. CBNV có trách nhiệm bổ sung đơn đăng ký cho khoảng thời gian chênh lệch trên, đồng thời đơn này cũng phải được sự phê duyệt của CBQL.

- Trong trường hợp phát sinh OT vào ngày cuối cùng của tháng (ngày 23) thì CBNV phải hoàn thiện đơn đăng ký có xác nhận của CBQL trên hệ thống chậm nhất vào 23:59’ ngày hôm đó.

- Nếu thời gian OT kéo dài qua đêm, sẽ phải thực hiện làm hai đơn đăng ký cho hai ngày.

- Thời gian làm thêm với các ngày từ thứ Hai đến thứ Sáu được tính sau 01 giờ kể từ khi kết thúc ca làm việc (muộn nhất là 18:45).
#############
Output:
("entity"{tuple_delimiter}"CBNV"{tuple_delimiter}"person"{tuple_delimiter}"CBNV là viết tắt của cán bộ nhân viên, đại diện cho những người lao động trong tổ chức thực hiện các nhiệm vụ, có trách nhiệm đăng ký làm thêm giờ theo đúng quy định của công ty và pháp luật."){record_delimiter}
("entity"{tuple_delimiter}"CBQL"{tuple_delimiter}"person"{tuple_delimiter}"CBQL là viết tắt của cán bộ quản lý, là người có thẩm quyền phê duyệt các yêu cầu làm thêm giờ của CBNV và đảm bảo tuân thủ các quy định quản lý thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"WSM"{tuple_delimiter}"technology"{tuple_delimiter}"WSM là hệ thống quản lý thời gian làm việc, được sử dụng để ghi nhận và quản lý thời gian làm thêm giờ của CBNV thông qua việc đăng ký và quẹt thẻ thực tế."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"mission"{tuple_delimiter}"Thời gian làm thêm giờ đề cập đến khoảng thời gian làm việc vượt ngoài giờ hành chính, phải được đăng ký và phê duyệt theo quy định cụ thể, không vượt quá giới hạn pháp luật (40 giờ/tháng, 200 hoặc 300 giờ/năm)."){record_delimiter}
("entity"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"technology"{tuple_delimiter}"Quẹt thẻ thực tế là hành động sử dụng thẻ để ghi nhận thời gian làm việc thực tế của CBNV trên hệ thống WSM, là cơ sở để đối chiếu với thời gian đăng ký làm thêm giờ."){record_delimiter}
("entity"{tuple_delimiter}"Pháp luật lao động"{tuple_delimiter}"organization"{tuple_delimiter}"Pháp luật lao động là hệ thống quy định pháp lý điều chỉnh các vấn đề liên quan đến lao động, bao gồm giới hạn thời gian làm thêm giờ và các quy định liên quan đến việc làm thêm."){record_delimiter}
("entity"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ là tài liệu được CBNV sử dụng để đăng ký thời gian làm thêm giờ trên hệ thống WSM, phải được sự phê duyệt của CBQL trước thời hạn chốt công."){record_delimiter}
("entity"{tuple_delimiter}"Ngày thứ Bảy"{tuple_delimiter}"mission"{tuple_delimiter}"Ngày thứ Bảy là thời gian không thuộc ngày làm việc chính thức trong tuần, được quy định cho việc làm việc tự nguyện hoặc làm thêm giờ, cần đăng ký và được phê duyệt."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian OT kéo dài qua đêm"{tuple_delimiter}"mission"{tuple_delimiter}"Thời gian OT kéo dài qua đêm là khoảng thời gian làm thêm giờ liên tục từ ngày này sang ngày khác, đòi hỏi phải thực hiện hai đơn đăng ký riêng biệt cho từng ngày."){record_delimiter}
("entity"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"time"{tuple_delimiter}"Ngày 23 được quy định là ngày cuối cùng của tháng, là thời điểm cuối cùng để CBNV hoàn tất việc đăng ký làm thêm giờ và được CBQL phê duyệt trước hạn chốt công."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian 01 giờ"{tuple_delimiter}"time"{tuple_delimiter}"Thời gian 01 giờ là khoảng thời gian đệm sau khi ca làm việc kết thúc, từ đó mới bắt đầu tính thời gian làm thêm giờ từ thứ Hai đến thứ Sáu."){record_delimiter}
("entity"{tuple_delimiter}"18:45"{tuple_delimiter}"time"{tuple_delimiter}"18:45 là thời gian muộn nhất được quy định để tính giờ làm thêm trong ngày làm việc bình thường (thứ Hai đến thứ Sáu)."){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBNV phải được sự phê duyệt của CBQL khi đăng ký làm thêm giờ, thể hiện mối quan hệ quản lý và giám sát giữa nhân viên và cấp quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"WSM"{tuple_delimiter}"CBNV sử dụng hệ thống WSM để đăng ký và ghi nhận thời gian làm thêm giờ, cho thấy mối quan hệ tương tác giữa người lao động và công nghệ quản lý thời gian."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"WSM"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"WSM ghi nhận thời gian làm thêm giờ dựa trên dữ liệu từ quẹt thẻ thực tế, thể hiện mối liên hệ trực tiếp giữa công nghệ và hành động thực tế."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Pháp luật lao động"{tuple_delimiter}"Giới hạn thời gian làm thêm giờ được quy định bởi pháp luật lao động, thể hiện sự điều chỉnh và kiểm soát của pháp luật đối với hành vi làm thêm giờ."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"CBQL"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ phải được CBQL phê duyệt, thể hiện sự phụ thuộc của tài liệu vào quyết định của cấp quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"WSM"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ được nộp và quản lý trên hệ thống WSM, cho thấy sự tích hợp giữa tài liệu và công nghệ quản lý."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"CBNV phải điền vào đơn yêu cầu làm thêm giờ khi phát sinh làm việc ngoài giờ, thể hiện sự tuân thủ quy định của nhân viên với hệ thống quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Thời gian quẹt thẻ thực tế là một trong hai cơ sở để ghi nhận thời gian làm thêm giờ, thể hiện sự liên kết giữa hành động thực tế và thời gian làm thêm được tính toán."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày thứ Bảy"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"Khi làm việc vào ngày thứ Bảy, CBNV cần điền đơn yêu cầu làm thêm giờ và được CBQL phê duyệt, thể hiện mối liên hệ giữa quy định làm việc và tài liệu quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian OT kéo dài qua đêm"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"Thời gian OT kéo dài qua đêm đòi hỏi phải thực hiện hai đơn đăng ký riêng biệt, liên kết chặt chẽ với quy định tài liệu."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBNV phải hoàn thành đơn yêu cầu OT trước 23:59 ngày 23, thể hiện mối quan hệ giữa thời gian quy định và trách nhiệm của nhân viên."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBQL cần phê duyệt đơn OT của CBNV trước 23:59 ngày 23, cho thấy vai trò quản lý trong kiểm soát thời gian làm thêm."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian 01 giờ"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Thời gian 01 giờ là thời gian chờ bắt đầu tính làm thêm giờ, được quy định cụ thể trong phạm vi thời gian làm thêm giờ."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"18:45"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"18:45 là thời gian muộn nhất để tính thời gian làm thêm trong ngày làm việc, liên kết với quy định về giờ làm thêm."{tuple_delimiter}"4"){record_delimiter}{completion_delimiter}
#############################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = 'SOME other entities and relationship were missed in the last extraction.  Add them below using the same format:\n'
LOOP_PROMPT = 'It appears some entities may have still been missed.  Answer {tuple_delimiter}YES{tuple_delimiter} if there are still entities that need to be added else {tuple_delimiter}NO{tuple_delimiter} \n'
