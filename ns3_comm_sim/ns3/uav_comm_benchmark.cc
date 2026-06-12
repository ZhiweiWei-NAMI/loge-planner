#include "ns3/applications-module.h"
#include "ns3/buildings-helper.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavCommBenchmark");

namespace
{

enum MessageType : uint8_t
{
    REQUEST_MESSAGE = 0,
    RESPONSE_MESSAGE = 1,
    FORWARD_MESSAGE = 2,
};

enum EventTypeCode : uint8_t
{
    PLAN_EVENT = 0,
    REPLAN_EVENT = 1,
    DECOUPLE_EVENT = 2,
};

enum EndpointCode : uint8_t
{
    EDGE_ENDPOINT = 0,
    CLOUD_ENDPOINT = 1,
    LOCAL_ENDPOINT = 2,
};

struct RegionSpec
{
    std::string regionId;
    std::string type;
    double xMin{0.0};
    double xMax{0.0};
    double yMin{0.0};
    double yMax{0.0};
    double zMin{0.0};
    double zMax{0.0};
    double extraLossDb{0.0};
    double extraJitterMs{0.0};
    double dropProbability{0.0};
    double buildingHeight{0.0};
};

struct EventRecord
{
    double time{0.0};
    uint32_t uavId{0};
    std::string eventType;
    uint32_t transactionId{0};
    uint32_t priority{0};
    uint32_t deadlineMs{0};
    uint32_t maxRetries{0};
    uint32_t requestPackets{1};
    uint32_t responsePackets{1};
};

struct TracePoint
{
    double time{0.0};
    Vector position;
};

std::ofstream gTxnFile;
std::ofstream gFlowFile;
std::ofstream gSnrFile;
std::string gArchitecture;
std::vector<Ptr<Building>> gBuildings;
std::vector<RegionSpec> gRegions;
std::map<uint16_t, Ipv4Address> gUavIpv4Map;

std::vector<std::string>
Split(const std::string& value, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter))
    {
        parts.push_back(item);
    }
    return parts;
}

std::string
Trim(const std::string& value)
{
    auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
    {
        return "";
    }
    auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

uint8_t
EndpointCodeFromString(const std::string& endpoint)
{
    if (endpoint == "cloud")
    {
        return CLOUD_ENDPOINT;
    }
    if (endpoint == "local")
    {
        return LOCAL_ENDPOINT;
    }
    return EDGE_ENDPOINT;
}

std::string
EndpointCodeToString(uint8_t endpoint)
{
    if (endpoint == CLOUD_ENDPOINT)
    {
        return "cloud";
    }
    if (endpoint == LOCAL_ENDPOINT)
    {
        return "local";
    }
    return "edge";
}

uint8_t
EventTypeFromString(const std::string& eventType)
{
    if (eventType == "replan")
    {
        return REPLAN_EVENT;
    }
    if (eventType == "decouple")
    {
        return DECOUPLE_EVENT;
    }
    return PLAN_EVENT;
}

std::string
EventTypeToString(uint8_t eventType)
{
    if (eventType == REPLAN_EVENT)
    {
        return "replan";
    }
    if (eventType == DECOUPLE_EVENT)
    {
        return "decouple";
    }
    return "plan";
}

std::vector<RegionSpec>
LoadRegions(const std::string& filePath)
{
    std::vector<RegionSpec> regions;
    std::ifstream input(filePath);
    if (!input.is_open())
    {
        NS_ABORT_MSG("Unable to open region file: " << filePath);
    }

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line))
    {
        if (Trim(line).empty())
        {
            continue;
        }
        auto cols = Split(line, ',');
        if (cols.size() < 11)
        {
            continue;
        }
        RegionSpec region;
        region.regionId = Trim(cols[0]);
        region.type = Trim(cols[1]);
        region.xMin = std::stod(cols[2]);
        region.xMax = std::stod(cols[3]);
        region.yMin = std::stod(cols[4]);
        region.yMax = std::stod(cols[5]);
        region.zMin = std::stod(cols[6]);
        region.zMax = std::stod(cols[7]);
        region.extraLossDb = std::stod(cols[8]);
        region.extraJitterMs = std::stod(cols[9]);
        region.dropProbability = std::stod(cols[10]);
        region.buildingHeight = cols.size() > 11 ? std::stod(cols[11]) : 0.0;
        regions.push_back(region);
    }
    return regions;
}

std::vector<EventRecord>
LoadEvents(const std::string& filePath)
{
    std::vector<EventRecord> events;
    std::ifstream input(filePath);
    if (!input.is_open())
    {
        NS_ABORT_MSG("Unable to open event file: " << filePath);
    }

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line))
    {
        if (Trim(line).empty())
        {
            continue;
        }
        auto cols = Split(line, ',');
        if (cols.size() < 7)
        {
            continue;
        }
        EventRecord event;
        event.time = std::stod(cols[0]);
        event.uavId = static_cast<uint32_t>(std::stoul(cols[1]));
        event.eventType = Trim(cols[2]);
        event.transactionId = static_cast<uint32_t>(std::stoul(cols[3]));
        event.priority = static_cast<uint32_t>(std::stoul(cols[4]));
        event.deadlineMs = static_cast<uint32_t>(std::stoul(cols[5]));
        event.maxRetries = static_cast<uint32_t>(std::stoul(cols[6]));
        event.requestPackets = cols.size() > 7 ? static_cast<uint32_t>(std::stoul(cols[7])) : 1u;
        event.responsePackets = cols.size() > 8 ? static_cast<uint32_t>(std::stoul(cols[8])) : 1u;
        events.push_back(event);
    }
    return events;
}

std::map<uint32_t, std::vector<TracePoint>>
LoadTraceMap(const std::string& filePath)
{
    std::map<uint32_t, std::vector<TracePoint>> traces;
    std::ifstream input(filePath);
    if (!input.is_open())
    {
        NS_ABORT_MSG("Unable to open trace file: " << filePath);
    }

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line))
    {
        if (Trim(line).empty())
        {
            continue;
        }
        auto cols = Split(line, ',');
        if (cols.size() < 5)
        {
            continue;
        }
        uint32_t uavId = static_cast<uint32_t>(std::stoul(cols[1]));
        TracePoint point;
        point.time = std::stod(cols[0]);
        point.position = Vector(std::stod(cols[2]), std::stod(cols[3]), std::stod(cols[4]));
        traces[uavId].push_back(point);
    }

    for (auto& [uavId, values] : traces)
    {
        std::sort(values.begin(), values.end(), [](const TracePoint& lhs, const TracePoint& rhs) {
            return lhs.time < rhs.time;
        });
    }
    return traces;
}

bool
PointInRegion(const Vector& point, const RegionSpec& region)
{
    return point.x >= region.xMin && point.x <= region.xMax && point.y >= region.yMin &&
           point.y <= region.yMax && point.z >= region.zMin && point.z <= region.zMax;
}

RegionSpec
LookupRegion(const Vector& point, const std::vector<RegionSpec>& regions)
{
    for (const auto& region : regions)
    {
        if (PointInRegion(point, region))
        {
            return region;
        }
    }
    return RegionSpec{"normal", "normal_region", 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0};
}

void
InstallTraceMobility(NodeContainer& uavNodes, const std::map<uint32_t, std::vector<TracePoint>>& traces)
{
    for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
    {
        auto mobility = CreateObject<WaypointMobilityModel>();
        auto it = traces.find(i);
        if (it == traces.end() || it->second.empty())
        {
            mobility->AddWaypoint(Waypoint(Seconds(0.0), Vector(0.0, 0.0, 20.0)));
        }
        else
        {
            for (const auto& point : it->second)
            {
                mobility->AddWaypoint(Waypoint(Seconds(point.time), point.position));
            }
        }
        uavNodes.Get(i)->AggregateObject(mobility);
    }
}

void
InstallConstantMobility(Ptr<Node> node, const Vector& position)
{
    auto mobility = CreateObject<ConstantPositionMobilityModel>();
    mobility->SetPosition(position);
    node->AggregateObject(mobility);
}

void
CreateBuildingsFromRegions(const std::vector<RegionSpec>& regions)
{
    for (const auto& region : regions)
    {
        if (region.buildingHeight <= 0.0)
        {
            continue;
        }
        auto building = CreateObject<Building>();
        building->SetBoundaries(
            Box(region.xMin, region.xMax, region.yMin, region.yMax, region.zMin, region.buildingHeight));
        building->SetBuildingType(Building::Commercial);
        building->SetExtWallsType(Building::ConcreteWithWindows);
        building->SetNRoomsX(1);
        building->SetNRoomsY(1);
        building->SetNFloors(1);
        gBuildings.push_back(building);
    }
}

class TransactionHeader : public Header
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TransactionHeader").SetParent<Header>().AddConstructor<TransactionHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 17;
    }

    void Serialize(Buffer::Iterator start) const override
    {
        start.WriteU8(m_messageType);
        start.WriteU8(m_eventType);
        start.WriteHtonU32(m_transactionId);
        start.WriteHtonU16(m_uavId);
        start.WriteU8(m_priority);
        start.WriteU8(m_attempt);
        start.WriteU8(m_packetIndex);
        start.WriteU8(m_packetCount);
        start.WriteU8(m_responsePacketCount);
        start.WriteHtonU16(m_deadlineMs);
        start.WriteU8(m_requestEndpoint);
        start.WriteU8(m_responseEndpoint);
    }

    uint32_t Deserialize(Buffer::Iterator start) override
    {
        m_messageType = start.ReadU8();
        m_eventType = start.ReadU8();
        m_transactionId = start.ReadNtohU32();
        m_uavId = start.ReadNtohU16();
        m_priority = start.ReadU8();
        m_attempt = start.ReadU8();
        m_packetIndex = start.ReadU8();
        m_packetCount = start.ReadU8();
        m_responsePacketCount = start.ReadU8();
        m_deadlineMs = start.ReadNtohU16();
        m_requestEndpoint = start.ReadU8();
        m_responseEndpoint = start.ReadU8();
        return GetSerializedSize();
    }

    void Print(std::ostream& os) const override
    {
        os << unsigned(m_messageType) << ":" << m_transactionId;
    }

    void SetMessageType(uint8_t value)
    {
        m_messageType = value;
    }

    void SetEventType(uint8_t value)
    {
        m_eventType = value;
    }

    void SetTransactionId(uint32_t value)
    {
        m_transactionId = value;
    }

    void SetUavId(uint16_t value)
    {
        m_uavId = value;
    }

    void SetPriority(uint8_t value)
    {
        m_priority = value;
    }

    void SetAttempt(uint8_t value)
    {
        m_attempt = value;
    }

    void SetPacketIndex(uint8_t value)
    {
        m_packetIndex = value;
    }

    void SetPacketCount(uint8_t value)
    {
        m_packetCount = value;
    }

    void SetResponsePacketCount(uint8_t value)
    {
        m_responsePacketCount = value;
    }

    void SetDeadlineMs(uint16_t value)
    {
        m_deadlineMs = value;
    }

    void SetResponseEndpoint(uint8_t value)
    {
        m_responseEndpoint = value;
    }

    void SetRequestEndpoint(uint8_t value)
    {
        m_requestEndpoint = value;
    }

    uint8_t GetMessageType() const
    {
        return m_messageType;
    }

    uint8_t GetEventType() const
    {
        return m_eventType;
    }

    uint32_t GetTransactionId() const
    {
        return m_transactionId;
    }

    uint16_t GetUavId() const
    {
        return m_uavId;
    }

    uint8_t GetPriority() const
    {
        return m_priority;
    }

    uint8_t GetAttempt() const
    {
        return m_attempt;
    }

    uint8_t GetPacketIndex() const
    {
        return m_packetIndex;
    }

    uint8_t GetPacketCount() const
    {
        return m_packetCount;
    }

    uint8_t GetResponsePacketCount() const
    {
        return m_responsePacketCount;
    }

    uint16_t GetDeadlineMs() const
    {
        return m_deadlineMs;
    }

    uint8_t GetResponseEndpoint() const
    {
        return m_responseEndpoint;
    }

    uint8_t GetRequestEndpoint() const
    {
        return m_requestEndpoint;
    }

  private:
    uint8_t m_messageType{0};
    uint8_t m_eventType{0};
    uint32_t m_transactionId{0};
    uint16_t m_uavId{0};
    uint8_t m_priority{0};
    uint8_t m_attempt{0};
    uint8_t m_packetIndex{0};
    uint8_t m_packetCount{0};
    uint8_t m_responsePacketCount{0};
    uint16_t m_deadlineMs{0};
    uint8_t m_requestEndpoint{EDGE_ENDPOINT};
    uint8_t m_responseEndpoint{EDGE_ENDPOINT};
};

void
LogTransaction(uint32_t txnId,
               uint32_t uavId,
               const std::string& eventType,
               bool success,
               bool stale,
               bool timeout,
               bool localExecution,
               bool networkRequired,
               uint32_t retryCount,
               double delayMs,
               uint32_t deadlineMs,
               const std::string& completionReason,
               const std::string& requestTarget,
               const std::string& responseSource,
               const std::string& regionLabel,
               uint32_t priority,
               const Vector& position)
{
    gTxnFile << std::fixed << std::setprecision(3) << Simulator::Now().GetSeconds() << ',' << txnId << ','
             << uavId << ',' << eventType << ',' << (success ? 1 : 0) << ',' << (stale ? 1 : 0) << ','
             << (timeout ? 1 : 0) << ',' << (localExecution ? 1 : 0) << ',' << (networkRequired ? 1 : 0)
             << ',' << retryCount << ',' << delayMs << ',' << deadlineMs << ',' << completionReason << ','
             << requestTarget << ',' << responseSource << ',' << regionLabel << ',' << priority << ','
             << position.x << ',' << position.y << ',' << position.z << ',' << gArchitecture << std::endl;
}

uint32_t
ParseNodeIdFromContext(const std::string& context)
{
    const std::string marker = "/NodeList/";
    auto start = context.find(marker);
    if (start == std::string::npos)
    {
        return std::numeric_limits<uint32_t>::max();
    }
    start += marker.size();
    auto end = context.find('/', start);
    if (end == std::string::npos)
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(std::stoul(context.substr(start, end - start)));
}

void
MonitorSniffRx(std::string context,
               Ptr<const Packet> packet,
               uint16_t channelFreqMhz,
               WifiTxVector txVector,
               MpduInfo aMpdu,
               SignalNoiseDbm signalNoise,
               uint16_t staId)
{
    double signal = static_cast<double>(signalNoise.signal);
    double noise = static_cast<double>(signalNoise.noise);
    uint32_t nodeId = ParseNodeIdFromContext(context);
    Vector position(0.0, 0.0, 0.0);
    std::string regionLabel = "normal_region";
    if (nodeId != std::numeric_limits<uint32_t>::max() && nodeId < NodeList::GetNNodes())
    {
        Ptr<Node> node = NodeList::GetNode(nodeId);
        Ptr<MobilityModel> mobility = node ? node->GetObject<MobilityModel>() : nullptr;
        if (mobility)
        {
            position = mobility->GetPosition();
            regionLabel = LookupRegion(position, gRegions).type;
        }
    }
    gSnrFile << std::fixed << std::setprecision(3) << Simulator::Now().GetSeconds() << ',' << context << ','
             << channelFreqMhz << ',' << signal << ',' << noise << ',' << (signal - noise) << ',' << staId
             << ',' << packet->GetSize() << ',' << nodeId << ',' << position.x << ',' << position.y << ','
             << position.z << ',' << regionLabel << std::endl;
}

class TxnServerApp : public Application
{
  public:
    void Setup(uint16_t port,
               uint16_t clientPort,
               uint8_t localEndpoint,
               Ipv4Address edgeAddress,
               Ipv4Address cloudAddress,
               uint32_t packetSize,
               double processingMs,
               double interPacketSpacingMs)
    {
        m_port = port;
        m_clientPort = clientPort;
        m_localEndpoint = localEndpoint;
        m_edgeAddress = edgeAddress;
        m_cloudAddress = cloudAddress;
        m_packetSize = packetSize;
        m_processingMs = processingMs;
        m_interPacketSpacingMs = interPacketSpacingMs;
    }

  private:
    struct RequestState
    {
        uint32_t receivedPackets{0};
        bool terminalActionScheduled{false};
        TransactionHeader header;
    };

    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_socket->SetRecvCallback(MakeCallback(&TxnServerApp::HandleRead, this));
    }

    void StopApplication() override
    {
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
        m_requests.clear();
    }

    std::string MakeKey(uint32_t txnId, uint8_t attempt) const
    {
        std::ostringstream builder;
        builder << txnId << ':' << unsigned(attempt);
        return builder.str();
    }

    void HandleRead(Ptr<Socket> socket)
    {
        Address from;
        while (auto packet = socket->RecvFrom(from))
        {
            TransactionHeader header;
            packet->RemoveHeader(header);
            if (header.GetMessageType() != REQUEST_MESSAGE && header.GetMessageType() != FORWARD_MESSAGE)
            {
                continue;
            }
            auto key = MakeKey(header.GetTransactionId(), header.GetAttempt());
            auto& state = m_requests[key];
            state.receivedPackets += 1;
            state.header = header;
            if (!state.terminalActionScheduled && state.receivedPackets >= header.GetPacketCount())
            {
                state.terminalActionScheduled = true;
                if (header.GetMessageType() == REQUEST_MESSAGE &&
                    header.GetResponseEndpoint() != m_localEndpoint)
                {
                    Simulator::ScheduleNow(&TxnServerApp::ForwardRequest, this, key);
                }
                else
                {
                    Simulator::Schedule(Seconds(m_processingMs / 1000.0),
                                        &TxnServerApp::SendResponses,
                                        this,
                                        key);
                }
            }
        }
    }

    Ipv4Address GetEndpointAddress(uint8_t endpoint) const
    {
        return endpoint == CLOUD_ENDPOINT ? m_cloudAddress : m_edgeAddress;
    }

    Ipv4Address GetUavAddress(uint16_t uavId) const
    {
        auto it = gUavIpv4Map.find(uavId);
        if (it == gUavIpv4Map.end())
        {
            return Ipv4Address::GetAny();
        }
        return it->second;
    }

    void ForwardRequest(const std::string& key)
    {
        auto it = m_requests.find(key);
        if (it == m_requests.end() || !m_socket)
        {
            return;
        }
        auto state = it->second;
        for (uint32_t index = 0; index < state.header.GetPacketCount(); ++index)
        {
            Simulator::Schedule(Seconds((index * m_interPacketSpacingMs) / 1000.0),
                                &TxnServerApp::SendOneForward,
                                this,
                                state.header,
                                index);
        }
        m_requests.erase(it);
    }

    void SendResponses(const std::string& key)
    {
        auto it = m_requests.find(key);
        if (it == m_requests.end() || !m_socket)
        {
            return;
        }
        auto state = it->second;
        for (uint32_t index = 0; index < state.header.GetResponsePacketCount(); ++index)
        {
            Simulator::Schedule(Seconds((index * m_interPacketSpacingMs) / 1000.0),
                                &TxnServerApp::SendOneResponse,
                                this,
                                state.header,
                                index);
        }
        m_requests.erase(it);
    }

    void SendOneForward(TransactionHeader requestHeader, uint32_t packetIndex)
    {
        if (!m_socket)
        {
            return;
        }
        TransactionHeader header = requestHeader;
        header.SetMessageType(FORWARD_MESSAGE);
        header.SetPacketIndex(static_cast<uint8_t>(packetIndex));
        uint32_t headerSize = header.GetSerializedSize();
        uint32_t payloadSize = m_packetSize > headerSize ? m_packetSize - headerSize : 0;
        auto packet = Create<Packet>(payloadSize);
        packet->AddHeader(header);
        m_socket->SendTo(packet,
                         0,
                         InetSocketAddress(GetEndpointAddress(requestHeader.GetResponseEndpoint()), m_port));
    }

    void SendOneResponse(TransactionHeader requestHeader, uint32_t packetIndex)
    {
        if (!m_socket)
        {
            return;
        }
        TransactionHeader header;
        header.SetMessageType(RESPONSE_MESSAGE);
        header.SetEventType(requestHeader.GetEventType());
        header.SetTransactionId(requestHeader.GetTransactionId());
        header.SetUavId(requestHeader.GetUavId());
        header.SetPriority(requestHeader.GetPriority());
        header.SetAttempt(requestHeader.GetAttempt());
        header.SetPacketIndex(static_cast<uint8_t>(packetIndex));
        header.SetPacketCount(requestHeader.GetResponsePacketCount());
        header.SetResponsePacketCount(requestHeader.GetResponsePacketCount());
        header.SetDeadlineMs(requestHeader.GetDeadlineMs());
        header.SetRequestEndpoint(requestHeader.GetRequestEndpoint());
        header.SetResponseEndpoint(requestHeader.GetResponseEndpoint());

        uint32_t headerSize = header.GetSerializedSize();
        uint32_t payloadSize = m_packetSize > headerSize ? m_packetSize - headerSize : 0;
        auto packet = Create<Packet>(payloadSize);
        packet->AddHeader(header);
        Ipv4Address uavAddress = GetUavAddress(requestHeader.GetUavId());
        if (uavAddress == Ipv4Address::GetAny())
        {
            return;
        }
        m_socket->SendTo(packet, 0, InetSocketAddress(uavAddress, m_clientPort));
    }

    Ptr<Socket> m_socket;
    uint16_t m_port{0};
    uint16_t m_clientPort{0};
    uint8_t m_localEndpoint{EDGE_ENDPOINT};
    Ipv4Address m_edgeAddress;
    Ipv4Address m_cloudAddress;
    uint32_t m_packetSize{0};
    double m_processingMs{0.0};
    double m_interPacketSpacingMs{1.0};
    std::map<std::string, RequestState> m_requests;
};

class TxnClientApp : public Application
{
  public:
    void Setup(const std::vector<EventRecord>& events,
               Ipv4Address edgeAddress,
               Ipv4Address cloudAddress,
               uint16_t port,
               uint16_t clientPort,
               uint32_t packetSize,
               uint32_t attemptTimeoutMs,
               uint32_t localProcessingMs,
               uint32_t interPacketSpacingMs,
               std::string planTarget,
               std::string replanTarget,
               std::string decoupleTarget,
               std::string planResponseSource,
               std::string replanResponseSource,
               std::string decoupleResponseSource,
               std::vector<RegionSpec> regions)
    {
        m_events = events;
        m_edgeAddress = edgeAddress;
        m_cloudAddress = cloudAddress;
        m_port = port;
        m_clientPort = clientPort;
        m_packetSize = packetSize;
        m_attemptTimeoutMs = attemptTimeoutMs;
        m_localProcessingMs = localProcessingMs;
        m_interPacketSpacingMs = interPacketSpacingMs;
        m_planTarget = std::move(planTarget);
        m_replanTarget = std::move(replanTarget);
        m_decoupleTarget = std::move(decoupleTarget);
        m_planResponseSource = std::move(planResponseSource);
        m_replanResponseSource = std::move(replanResponseSource);
        m_decoupleResponseSource = std::move(decoupleResponseSource);
        m_regions = std::move(regions);
    }

  private:
    struct PendingTxn
    {
        EventRecord event;
        bool networkRequired{true};
        uint32_t retryCount{0};
        uint32_t currentAttempt{0};
        uint32_t receivedResponsePackets{0};
        uint32_t expectedResponsePackets{1};
        std::string requestTarget;
        std::string responseSource;
        std::string regionLabel;
        double regionExtraJitterMs{0.0};
        double regionDropProbability{0.0};
        Vector position;
        Time startTime;
        Time deadlineTime;
        EventId timeoutEvent;
    };

    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_clientPort));
        m_socket->SetRecvCallback(MakeCallback(&TxnClientApp::HandleRead, this));
        m_uniform = CreateObject<UniformRandomVariable>();
        for (const auto& event : m_events)
        {
            Simulator::Schedule(Seconds(event.time), &TxnClientApp::StartTransaction, this, event);
        }
    }

    void StopApplication() override
    {
        for (auto& [txnId, pending] : m_pending)
        {
            if (pending.timeoutEvent.IsRunning())
            {
                pending.timeoutEvent.Cancel();
            }
            double delayMs = (Simulator::Now() - pending.startTime).GetMilliSeconds();
            LogTransaction(txnId,
                           pending.event.uavId,
                           pending.event.eventType,
                           false,
                           false,
                           true,
                           false,
                           pending.networkRequired,
                           pending.retryCount,
                           delayMs,
                           pending.event.deadlineMs,
                           "simulation_end",
                           pending.requestTarget,
                           pending.responseSource,
                           pending.regionLabel,
                           pending.event.priority,
                           pending.position);
        }
        m_pending.clear();
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    std::string DetermineTarget(const EventRecord& event) const
    {
        if (event.eventType == "replan")
        {
            return m_replanTarget;
        }
        if (event.eventType == "decouple")
        {
            return m_decoupleTarget;
        }
        return m_planTarget;
    }

    std::string DetermineResponseSource(const EventRecord& event) const
    {
        if (event.eventType == "replan")
        {
            return m_replanResponseSource;
        }
        if (event.eventType == "decouple")
        {
            return m_decoupleResponseSource;
        }
        return m_planResponseSource;
    }

    uint32_t DetermineRequestPacketCount(const EventRecord& event) const
    {
        return event.requestPackets;
    }

    void CompleteLocalTransaction(EventRecord event, std::string regionLabel, Vector position)
    {
        double delayMs = static_cast<double>(m_localProcessingMs);
        bool stale = delayMs > static_cast<double>(event.deadlineMs);
        LogTransaction(event.transactionId,
                       event.uavId,
                       event.eventType,
                       true,
                       stale,
                       false,
                       true,
                       false,
                       0,
                       delayMs,
                       event.deadlineMs,
                       stale ? "local_stale" : "local_execution",
                       "local",
                       "local",
                       regionLabel,
                       event.priority,
                       position);
    }

    void StartTransaction(EventRecord event)
    {
        Ptr<MobilityModel> mobility = GetNode()->GetObject<MobilityModel>();
        Vector position = mobility ? mobility->GetPosition() : Vector(0.0, 0.0, 0.0);
        RegionSpec region = LookupRegion(position, m_regions);
        std::string requestTarget = DetermineTarget(event);
        std::string responseSource = DetermineResponseSource(event);

        if (requestTarget == "local" || responseSource == "local")
        {
            Simulator::Schedule(Seconds(m_localProcessingMs / 1000.0),
                                &TxnClientApp::CompleteLocalTransaction,
                                this,
                                event,
                                region.type,
                                position);
            return;
        }

        PendingTxn pending;
        pending.event = event;
        pending.expectedResponsePackets = event.responsePackets;
        pending.requestTarget = requestTarget;
        pending.responseSource = responseSource;
        pending.regionLabel = region.type;
        pending.regionExtraJitterMs = region.extraJitterMs;
        pending.regionDropProbability = region.dropProbability;
        pending.position = position;
        pending.startTime = Simulator::Now();
        pending.deadlineTime = Simulator::Now() + MilliSeconds(event.deadlineMs);
        m_pending[event.transactionId] = pending;
        SendAttempt(event.transactionId, 0);
    }

    void SendAttempt(uint32_t transactionId, uint32_t attempt)
    {
        auto it = m_pending.find(transactionId);
        if (it == m_pending.end())
        {
            return;
        }
        auto& pending = it->second;
        pending.currentAttempt = attempt;
        pending.receivedResponsePackets = 0;

        uint32_t packetCount = DetermineRequestPacketCount(pending.event);
        for (uint32_t index = 0; index < packetCount; ++index)
        {
            double jitter = pending.regionExtraJitterMs > 0.0 ? m_uniform->GetValue(0.0, pending.regionExtraJitterMs) : 0.0;
            Simulator::Schedule(Seconds((index * m_interPacketSpacingMs + jitter) / 1000.0),
                                &TxnClientApp::SendOnePacket,
                                this,
                                transactionId,
                                attempt,
                                index,
                                packetCount);
        }
        pending.timeoutEvent = Simulator::Schedule(Seconds((m_attemptTimeoutMs + pending.regionExtraJitterMs) / 1000.0),
                                                   &TxnClientApp::HandleTimeout,
                                                   this,
                                                   transactionId,
                                                   attempt);
    }

    void SendOnePacket(uint32_t transactionId, uint32_t attempt, uint32_t packetIndex, uint32_t packetCount)
    {
        auto it = m_pending.find(transactionId);
        if (it == m_pending.end() || !m_socket)
        {
            return;
        }
        auto& pending = it->second;
        if (attempt != pending.currentAttempt)
        {
            return;
        }
        if (m_uniform->GetValue() < pending.regionDropProbability)
        {
            return;
        }

        TransactionHeader header;
        header.SetMessageType(REQUEST_MESSAGE);
        header.SetEventType(EventTypeFromString(pending.event.eventType));
        header.SetTransactionId(transactionId);
        header.SetUavId(static_cast<uint16_t>(pending.event.uavId));
        header.SetPriority(static_cast<uint8_t>(pending.event.priority));
        header.SetAttempt(static_cast<uint8_t>(attempt));
        header.SetPacketIndex(static_cast<uint8_t>(packetIndex));
        header.SetPacketCount(static_cast<uint8_t>(packetCount));
        header.SetResponsePacketCount(static_cast<uint8_t>(pending.event.responsePackets));
        header.SetDeadlineMs(static_cast<uint16_t>(pending.event.deadlineMs));
        header.SetRequestEndpoint(EndpointCodeFromString(pending.requestTarget));
        header.SetResponseEndpoint(EndpointCodeFromString(pending.responseSource));

        uint32_t headerSize = header.GetSerializedSize();
        uint32_t payloadSize = m_packetSize > headerSize ? m_packetSize - headerSize : 0;
        auto packet = Create<Packet>(payloadSize);
        packet->AddHeader(header);
        Ipv4Address address = pending.requestTarget == "cloud" ? m_cloudAddress : m_edgeAddress;
        m_socket->SendTo(packet, 0, InetSocketAddress(address, m_port));
    }

    void HandleTimeout(uint32_t transactionId, uint32_t attempt)
    {
        auto it = m_pending.find(transactionId);
        if (it == m_pending.end())
        {
            return;
        }
        auto pending = it->second;
        if (attempt != pending.currentAttempt)
        {
            return;
        }
        if (Simulator::Now() > pending.deadlineTime || pending.retryCount >= pending.event.maxRetries)
        {
            LogTransaction(transactionId,
                           pending.event.uavId,
                           pending.event.eventType,
                           false,
                           false,
                           true,
                           false,
                           pending.networkRequired,
                           pending.retryCount,
                           (Simulator::Now() - pending.startTime).GetMilliSeconds(),
                           pending.event.deadlineMs,
                           "timeout",
                           pending.requestTarget,
                           pending.responseSource,
                           pending.regionLabel,
                           pending.event.priority,
                           pending.position);
            m_pending.erase(it);
            return;
        }

        it->second.retryCount += 1;
        SendAttempt(transactionId, attempt + 1);
    }

    void HandleRead(Ptr<Socket> socket)
    {
        Address from;
        while (auto packet = socket->RecvFrom(from))
        {
            TransactionHeader header;
            packet->RemoveHeader(header);
            if (header.GetMessageType() != RESPONSE_MESSAGE)
            {
                continue;
            }
            auto it = m_pending.find(header.GetTransactionId());
            if (it == m_pending.end())
            {
                continue;
            }
            auto& pending = it->second;
            if (header.GetAttempt() != pending.currentAttempt)
            {
                continue;
            }
            pending.receivedResponsePackets += 1;
            if (pending.receivedResponsePackets >= pending.expectedResponsePackets)
            {
                if (pending.timeoutEvent.IsRunning())
                {
                    pending.timeoutEvent.Cancel();
                }
                double delayMs = (Simulator::Now() - pending.startTime).GetMilliSeconds();
                bool stale = Simulator::Now() > pending.deadlineTime;
                LogTransaction(header.GetTransactionId(),
                               pending.event.uavId,
                               pending.event.eventType,
                               !stale,
                               stale,
                               false,
                               false,
                               pending.networkRequired,
                               pending.retryCount,
                               delayMs,
                               pending.event.deadlineMs,
                               stale ? "stale_response" : "response",
                               pending.requestTarget,
                               pending.responseSource,
                               pending.regionLabel,
                               pending.event.priority,
                               pending.position);
                m_pending.erase(it);
            }
        }
    }

    Ptr<Socket> m_socket;
    Ptr<UniformRandomVariable> m_uniform;
    std::vector<EventRecord> m_events;
    Ipv4Address m_edgeAddress;
    Ipv4Address m_cloudAddress;
    uint16_t m_port{0};
    uint16_t m_clientPort{0};
    uint32_t m_packetSize{0};
    uint32_t m_attemptTimeoutMs{0};
    uint32_t m_localProcessingMs{0};
    uint32_t m_interPacketSpacingMs{0};
    std::string m_planTarget{"cloud"};
    std::string m_replanTarget{"edge"};
    std::string m_decoupleTarget{"cloud"};
    std::string m_planResponseSource{"cloud"};
    std::string m_replanResponseSource{"edge"};
    std::string m_decoupleResponseSource{"cloud"};
    std::vector<RegionSpec> m_regions;
    std::map<uint32_t, PendingTxn> m_pending;
};

void
WriteFlowSummary(Ptr<FlowMonitor> monitor, FlowMonitorHelper& flowHelper)
{
    monitor->CheckForLostPackets();
    auto classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    auto stats = monitor->GetFlowStats();
    for (const auto& [flowId, stat] : stats)
    {
        auto tuple = classifier->FindFlow(flowId);
        double meanDelayMs = stat.rxPackets > 0 ? stat.delaySum.GetMilliSeconds() / stat.rxPackets : 0.0;
        double meanJitterMs = stat.rxPackets > 1 ? stat.jitterSum.GetMilliSeconds() / (stat.rxPackets - 1) : 0.0;
        gFlowFile << flowId << ',' << tuple.sourceAddress << ',' << tuple.destinationAddress << ','
                  << tuple.sourcePort << ',' << tuple.destinationPort << ',' << stat.txPackets << ','
                  << stat.rxPackets << ',' << stat.lostPackets << ',' << stat.txBytes << ',' << stat.rxBytes
                  << ',' << meanDelayMs << ',' << meanJitterMs << std::endl;
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string traceFile;
    std::string eventFile;
    std::string regionFile;
    std::string outputDir = ".";
    std::string architecture = "cloud";
    std::string phyMode = "HtMcs3";
    std::string controlMode = "HtMcs0";
    std::string planTarget = "cloud";
    std::string replanTarget = "edge";
    std::string decoupleTarget = "cloud";
    std::string planResponseSource = "cloud";
    std::string replanResponseSource = "edge";
    std::string decoupleResponseSource = "cloud";
    double simulationTime = 30.0;
    double txPowerDbm = 18.0;
    double frequencyHz = 5e9;
    uint32_t channelWidthMHz = 20;
    double backhaulDelayMs = 12.0;
    double backhaulRateMbps = 1000.0;
    double backhaulPacketErrorRate = 0.0;
    double edgeApHeight = 25.0;
    double areaWidth = 1200.0;
    double areaHeight = 800.0;
    uint32_t packetSize = 1024;
    uint32_t attemptTimeoutMs = 120;
    uint32_t localProcessingMs = 8;
    uint32_t edgeProcessingMs = 20;
    uint32_t cloudProcessingMs = 60;
    uint32_t interRequestSpacingMs = 5;
    uint32_t backgroundStaCount = 0;
    double backgroundContentionLoadMbps = 0.0;
    uint32_t enablePcap = 0;
    uint32_t enableNetAnim = 0;
    uint32_t enableNetAnimPacketMetadata = 0;
    double netAnimMobilityPollIntervalMs = 500.0;
    uint32_t seed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("traceFile", "Trace CSV file", traceFile);
    cmd.AddValue("eventFile", "Event CSV file", eventFile);
    cmd.AddValue("regionFile", "Region CSV file", regionFile);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.AddValue("architecture", "Architecture name", architecture);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("packetSize", "Packet size in bytes", packetSize);
    cmd.AddValue("attemptTimeoutMs", "Attempt timeout in milliseconds", attemptTimeoutMs);
    cmd.AddValue("localProcessingMs", "Local processing delay in ms", localProcessingMs);
    cmd.AddValue("edgeProcessingMs", "Edge processing delay in ms", edgeProcessingMs);
    cmd.AddValue("cloudProcessingMs", "Cloud processing delay in ms", cloudProcessingMs);
    cmd.AddValue("interRequestSpacingMs", "Inter-packet spacing in ms", interRequestSpacingMs);
    cmd.AddValue("phyMode", "Wifi data mode", phyMode);
    cmd.AddValue("controlMode", "Wifi control mode", controlMode);
    cmd.AddValue("txPowerDbm", "Wifi tx power in dBm", txPowerDbm);
    cmd.AddValue("frequencyHz", "Wifi frequency in Hz", frequencyHz);
    cmd.AddValue("channelWidthMHz", "Wifi channel width in MHz", channelWidthMHz);
    cmd.AddValue("backhaulDelayMs", "Edge-cloud backhaul delay in ms", backhaulDelayMs);
    cmd.AddValue("backhaulRateMbps", "Edge-cloud backhaul rate in Mbps", backhaulRateMbps);
    cmd.AddValue("backhaulPacketErrorRate",
                 "Edge-cloud backhaul packet error rate",
                 backhaulPacketErrorRate);
    cmd.AddValue("edgeApHeight", "Edge AP height", edgeApHeight);
    cmd.AddValue("areaWidth", "Scenario width", areaWidth);
    cmd.AddValue("areaHeight", "Scenario height", areaHeight);
    cmd.AddValue("backgroundStaCount", "Background STA count", backgroundStaCount);
    cmd.AddValue("backgroundContentionLoadMbps",
                 "Aggregate background contention load in Mbps",
                 backgroundContentionLoadMbps);
    cmd.AddValue("planTarget", "Target for plan tasks", planTarget);
    cmd.AddValue("replanTarget", "Target for replan tasks", replanTarget);
    cmd.AddValue("decoupleTarget", "Target for decouple tasks", decoupleTarget);
    cmd.AddValue("planResponseSource", "Response source for plan tasks", planResponseSource);
    cmd.AddValue("replanResponseSource", "Response source for replan tasks", replanResponseSource);
    cmd.AddValue("decoupleResponseSource", "Response source for decouple tasks", decoupleResponseSource);
    cmd.AddValue("enablePcap", "Enable pcap output", enablePcap);
    cmd.AddValue("enableNetAnim", "Enable NetAnim XML output", enableNetAnim);
    cmd.AddValue("enableNetAnimPacketMetadata",
                 "Enable NetAnim packet metadata",
                 enableNetAnimPacketMetadata);
    cmd.AddValue("netAnimMobilityPollIntervalMs",
                 "NetAnim mobility poll interval in ms",
                 netAnimMobilityPollIntervalMs);
    cmd.AddValue("seed", "Random seed", seed);
    cmd.Parse(argc, argv);

    SeedManager::SetSeed(seed);
    SeedManager::SetRun(1);

    NS_ABORT_MSG_IF(traceFile.empty(), "traceFile is required");
    NS_ABORT_MSG_IF(eventFile.empty(), "eventFile is required");
    NS_ABORT_MSG_IF(regionFile.empty(), "regionFile is required");

    auto traces = LoadTraceMap(traceFile);
    auto events = LoadEvents(eventFile);
    auto regions = LoadRegions(regionFile);
    gRegions = regions;

    uint32_t uavCount = static_cast<uint32_t>(traces.size());
    NodeContainer edgeNode;
    edgeNode.Create(1);
    NodeContainer cloudNode;
    cloudNode.Create(1);
    NodeContainer uavNodes;
    uavNodes.Create(uavCount);
    NodeContainer backgroundNodes;
    backgroundNodes.Create(backgroundStaCount);

    InstallTraceMobility(uavNodes, traces);
    InstallConstantMobility(edgeNode.Get(0), Vector(areaWidth / 2.0, areaHeight / 2.0, edgeApHeight));
    InstallConstantMobility(cloudNode.Get(0), Vector(areaWidth / 2.0 + 10.0, areaHeight / 2.0 + 10.0, edgeApHeight + 5.0));
    const double pi = 3.14159265358979323846;
    for (uint32_t i = 0; i < backgroundNodes.GetN(); ++i)
    {
        double angle = (2.0 * pi * i) / std::max(1u, backgroundNodes.GetN());
        InstallConstantMobility(backgroundNodes.Get(i),
                                Vector(areaWidth * 0.72 + 40.0 * std::cos(angle),
                                       areaHeight * 0.5 + 40.0 * std::sin(angle),
                                       2.0));
    }

    CreateBuildingsFromRegions(regions);

    NodeContainer allWirelessNodes;
    allWirelessNodes.Add(edgeNode);
    allWirelessNodes.Add(uavNodes);
    allWirelessNodes.Add(backgroundNodes);
    BuildingsHelper::Install(allWirelessNodes);
    BuildingsHelper::Install(cloudNode);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(phyMode),
                                 "ControlMode",
                                 StringValue(controlMode));

    YansWifiChannelHelper channelHelper;
    channelHelper.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channelHelper.AddPropagationLoss("ns3::HybridBuildingsPropagationLossModel",
                                     "Frequency",
                                     DoubleValue(frequencyHz),
                                     "Environment",
                                     EnumValue(UrbanEnvironment),
                                     "CitySize",
                                     EnumValue(LargeCity),
                                     "RooftopLevel",
                                     DoubleValue(edgeApHeight),
                                     "ShadowSigmaOutdoor",
                                     DoubleValue(0.0),
                                     "ShadowSigmaIndoor",
                                     DoubleValue(0.0),
                                     "ShadowSigmaExtWalls",
                                     DoubleValue(0.0));

    YansWifiPhyHelper phy;
    phy.SetChannel(channelHelper.Create());
    phy.Set("TxPowerStart", DoubleValue(txPowerDbm));
    phy.Set("TxPowerEnd", DoubleValue(txPowerDbm));
    std::ostringstream channelSettings;
    channelSettings << "{0, " << channelWidthMHz << ", "
                    << (frequencyHz >= 5e9 ? "BAND_5GHZ" : "BAND_2_4GHZ") << ", 0}";
    phy.Set("ChannelSettings", StringValue(channelSettings.str()));
    phy.SetErrorRateModel("ns3::YansErrorRateModel");

    WifiMacHelper mac;
    Ssid ssid = Ssid("UavControlNet");
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, edgeNode);

    NodeContainer staNodes;
    staNodes.Add(uavNodes);
    staNodes.Add(backgroundNodes);
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, staNodes);

    PointToPointHelper p2p;
    std::ostringstream dataRateStream;
    dataRateStream << static_cast<uint64_t>(backhaulRateMbps) << "Mbps";
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRateStream.str()));
    p2p.SetChannelAttribute("Delay", TimeValue(Seconds(backhaulDelayMs / 1000.0)));
    NetDeviceContainer backhaulDevices = p2p.Install(edgeNode.Get(0), cloudNode.Get(0));
    if (backhaulPacketErrorRate > 0.0)
    {
        Ptr<RateErrorModel> edgeBackhaulRxError = CreateObject<RateErrorModel>();
        edgeBackhaulRxError->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
        edgeBackhaulRxError->SetRate(backhaulPacketErrorRate);
        backhaulDevices.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(edgeBackhaulRxError));

        Ptr<RateErrorModel> cloudBackhaulRxError = CreateObject<RateErrorModel>();
        cloudBackhaulRxError->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
        cloudBackhaulRxError->SetRate(backhaulPacketErrorRate);
        backhaulDevices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(cloudBackhaulRxError));
    }

    InternetStackHelper stack;
    stack.Install(edgeNode);
    stack.Install(cloudNode);
    stack.Install(uavNodes);
    stack.Install(backgroundNodes);

    Ipv4AddressHelper wifiAddress;
    wifiAddress.SetBase("10.1.0.0", "255.255.255.0");
    NetDeviceContainer wifiDevices;
    wifiDevices.Add(apDevice);
    wifiDevices.Add(staDevices);
    Ipv4InterfaceContainer wifiInterfaces = wifiAddress.Assign(wifiDevices);
    Ipv4Address edgeWifiAddress = wifiInterfaces.GetAddress(0);

    Ipv4AddressHelper backhaulAddress;
    backhaulAddress.SetBase("10.2.0.0", "255.255.255.0");
    Ipv4InterfaceContainer backhaulInterfaces = backhaulAddress.Assign(backhaulDevices);
    Ipv4Address cloudAddress = backhaulInterfaces.GetAddress(1);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    uint16_t txnPort = 9000;
    uint16_t clientPort = 9100;
    uint16_t backgroundPort = 9001;

    gUavIpv4Map.clear();
    for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
    {
        gUavIpv4Map[static_cast<uint16_t>(i)] = wifiInterfaces.GetAddress(1 + i);
    }

    auto edgeServer = CreateObject<TxnServerApp>();
    edgeServer->Setup(txnPort,
                      clientPort,
                      EDGE_ENDPOINT,
                      edgeWifiAddress,
                      cloudAddress,
                      packetSize,
                      edgeProcessingMs,
                      interRequestSpacingMs);
    edgeNode.Get(0)->AddApplication(edgeServer);
    edgeServer->SetStartTime(Seconds(0.0));
    edgeServer->SetStopTime(Seconds(simulationTime + 2.0));

    auto cloudServer = CreateObject<TxnServerApp>();
    cloudServer->Setup(txnPort,
                       clientPort,
                       CLOUD_ENDPOINT,
                       edgeWifiAddress,
                       cloudAddress,
                       packetSize,
                       cloudProcessingMs,
                       interRequestSpacingMs);
    cloudNode.Get(0)->AddApplication(cloudServer);
    cloudServer->SetStartTime(Seconds(0.0));
    cloudServer->SetStopTime(Seconds(simulationTime + 2.0));

    PacketSinkHelper backgroundSink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), backgroundPort));
    ApplicationContainer backgroundSinkApp = backgroundSink.Install(edgeNode.Get(0));
    backgroundSinkApp.Start(Seconds(0.0));
    backgroundSinkApp.Stop(Seconds(simulationTime + 2.0));

    double perBackgroundStaLoadMbps =
        backgroundStaCount > 0 ? backgroundContentionLoadMbps / static_cast<double>(backgroundStaCount) : 0.0;
    for (uint32_t i = 0; i < backgroundNodes.GetN(); ++i)
    {
        if (perBackgroundStaLoadMbps > 0.0)
        {
            std::ostringstream rateStream;
            rateStream << perBackgroundStaLoadMbps << "Mbps";
            OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(edgeWifiAddress, backgroundPort));
            onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            onoff.SetAttribute("DataRate", DataRateValue(DataRate(rateStream.str())));
            auto apps = onoff.Install(backgroundNodes.Get(i));
            apps.Start(Seconds(1.0 + 0.05 * i));
            apps.Stop(Seconds(simulationTime + 1.0));
        }
    }

    std::map<uint32_t, std::vector<EventRecord>> eventsByUav;
    for (const auto& event : events)
    {
        eventsByUav[event.uavId].push_back(event);
    }

    for (uint32_t uavId = 0; uavId < uavNodes.GetN(); ++uavId)
    {
        auto client = CreateObject<TxnClientApp>();
        client->Setup(eventsByUav[uavId],
                      edgeWifiAddress,
                      cloudAddress,
                      txnPort,
                      clientPort,
                      packetSize,
                      attemptTimeoutMs,
                      localProcessingMs,
                      interRequestSpacingMs,
                      planTarget,
                      replanTarget,
                      decoupleTarget,
                      planResponseSource,
                      replanResponseSource,
                      decoupleResponseSource,
                      regions);
        uavNodes.Get(uavId)->AddApplication(client);
        client->SetStartTime(Seconds(0.0));
        client->SetStopTime(Seconds(simulationTime + 2.0));
    }

    if (enablePcap != 0)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap(outputDir + "/wifi", wifiDevices);
    }

    std::unique_ptr<AnimationInterface> anim;
    if (enableNetAnim != 0)
    {
        anim = std::make_unique<AnimationInterface>(outputDir + "/netanim.xml");
        anim->SetMobilityPollInterval(Seconds(netAnimMobilityPollIntervalMs / 1000.0));
        if (enableNetAnimPacketMetadata != 0)
        {
            anim->EnablePacketMetadata(true);
        }

        anim->UpdateNodeDescription(edgeNode.Get(0), "edge");
        anim->UpdateNodeColor(edgeNode.Get(0), 0, 180, 0);
        anim->UpdateNodeDescription(cloudNode.Get(0), "cloud");
        anim->UpdateNodeColor(cloudNode.Get(0), 0, 80, 255);
        for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
        {
            anim->UpdateNodeDescription(uavNodes.Get(i), "uav-" + std::to_string(i));
            anim->UpdateNodeColor(uavNodes.Get(i), 220, 30, 30);
        }
        for (uint32_t i = 0; i < backgroundNodes.GetN(); ++i)
        {
            anim->UpdateNodeDescription(backgroundNodes.Get(i), "bg-" + std::to_string(i));
            anim->UpdateNodeColor(backgroundNodes.Get(i), 120, 120, 120);
        }
    }

    gArchitecture = architecture;
    gTxnFile.open(outputDir + "/transactions.csv");
    gTxnFile << "sim_time_s,transaction_id,uav_id,event_type,success,stale,timeout,local_execution,network_required,retry_count,critical_path_delay_ms,deadline_ms,completion_reason,request_target,response_source,region_label,priority,x,y,z,architecture" << std::endl;
    gFlowFile.open(outputDir + "/flow_summary.csv");
    gFlowFile << "flow_id,source_address,destination_address,source_port,destination_port,tx_packets,rx_packets,lost_packets,tx_bytes,rx_bytes,mean_delay_ms,mean_jitter_ms" << std::endl;
    gSnrFile.open(outputDir + "/snr_samples.csv");
    gSnrFile << "sim_time_s,context,channel_freq_mhz,signal_dbm,noise_dbm,snr_db,sta_id,packet_size,node_id,x,y,z,region_label" << std::endl;

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
                    MakeCallback(&MonitorSniffRx));

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    Simulator::Stop(Seconds(simulationTime + 2.0));
    Simulator::Run();

    WriteFlowSummary(monitor, flowHelper);
    flowHelper.SerializeToXmlFile(outputDir + "/flow_monitor.xml", false, false);

    Simulator::Destroy();
    gTxnFile.close();
    gFlowFile.close();
    gSnrFile.close();
    return 0;
}
